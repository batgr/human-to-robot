import io

import numpy as np
import pytest
import torch
from PIL import Image

from h2r.models.controller import Controller
from h2r.training.controller import controller_loss, encode_episodes, evaluate

N, D, A = 16, 32, 7


class StubWorldModel:
    """encode: one embedding per frame from its mean pixel; idm: difference of consecutive embeddings"""

    def encode(self, info):
        px = info["pixels"].float()
        return {"emb": px.mean(dim=(2, 3, 4))[..., None, None].expand(-1, -1, N, D)}

    def idm(self, emb):
        return emb[:, 1:, 0] - emb[:, :-1, 0], torch.tensor(0.0)


def jpeg(value):
    buf = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), value, np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def episode(n_frames, n_actions):
    return {"video": "ep", "frames": [jpeg(10 * i) for i in range(n_frames)], "actions": [[0.1] * A] * n_actions}


def test_encode_episodes():
    data = encode_episodes(StubWorldModel(), [episode(5, 4), episode(3, 2)], "cpu")
    assert data["s"].shape == (6, N, D) and data["s"].dtype == torch.float16
    assert data["a"].shape == (6, A) and data["mu"].shape == (6, D)
    assert torch.allclose(data["mu"][0], data["s"][1, 0].float() - data["s"][0, 0].float(), atol=0.1)


def test_frames_and_actions_must_match():
    with pytest.raises(ValueError):
        encode_episodes(StubWorldModel(), [episode(5, 5)], "cpu")


def test_learns_a_latent_action():
    """mu depends on the action and on where the gripper is (patch 0 or 1): the controller must use both"""
    torch.manual_seed(0)
    gripper = torch.randn(D)  # a constant offset would be removed by LayerNorm

    def make(m):
        s, a = torch.zeros(m, N, D), torch.randn(m, A)
        where = torch.randint(0, 2, (m,))
        s[torch.arange(m), where] = gripper
        mu = torch.zeros(m, D)
        mu[torch.arange(m), where] = a[:, 0]
        return {"s": s.half(), "a": a, "mu": mu}

    train, val = make(2048), make(256)
    ctrl = Controller(num_patches=N, dim=D, act_dim=A)
    ctrl.set_action_stats(train["a"])
    autocast = torch.autocast("cpu", enabled=False)
    start = evaluate(ctrl, val, "cpu", autocast)
    opt = torch.optim.AdamW(ctrl.parameters(), lr=1e-3)
    for _ in range(300):
        idx = torch.randint(len(train["a"]), (128,))
        loss = controller_loss(ctrl, train["s"][idx], train["a"][idx], train["mu"][idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert evaluate(ctrl, val, "cpu", autocast) < 0.2 * start
