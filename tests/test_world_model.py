import pytest
import torch
import torch.nn.functional as F
from test_jepa import small_model


def test_rollout_terms_match_an_explicit_unrolling():
    """L_k = error on state j after k steps from the real state j - k, each step in a window of w states
    that holds the previous predictions; every j >= k is a target"""
    from h2r.training.world_model import rollout_terms

    model, D = small_model()
    T, w = 7, 3
    emb, z = torch.randn(2, T, D), torch.randn(2, T - 1, D)

    def unroll(j, k):
        """prediction of state j after k steps from j - k, windows clipped at the start of the clip"""
        states = {p: emb[:, p] for p in range(j - k + 1)}
        for q in range(j - k + 1, j + 1):  # predict state q from the window before it
            a = max(0, q - w)
            ctx = torch.stack([states[p] for p in range(a, q)], dim=1)
            states[q] = model.predict(ctx, z[:, a:q])[:, -1]
        return states[j]

    terms = rollout_terms(model, emb, z, steps=4, w=w)
    assert len(terms) == 3  # L_2, L_3, L_4
    for k, term in enumerate(terms, start=2):
        expected = torch.stack([unroll(j, k) for j in range(k, T)], dim=1)
        assert torch.allclose(term, F.mse_loss(expected, emb[:, k:]), atol=1e-5)
    assert rollout_terms(model, emb, z, steps=1, w=w) == []


def test_stop_grad_target_cuts_only_the_target_path():
    """loss.stop_grad_target: same loss value; no gradient of tf reaches a state used only as a target (the IDM's z
    held fixed here), while the states read by the predictor still get one"""
    from h2r.models.module import SIGReg
    from h2r.training.world_model import losses

    model, D = small_model()
    z = torch.randn(2, 3, D)
    model.idm.forward = lambda emb: (z, torch.zeros(()))  # z without a path to the states
    cfg = {"loss": {"beta": 0.0, "sigreg": 0.0, "rollout_steps": 1, "rollout_window": 3}}
    emb = torch.randn(2, 4, D, requires_grad=True)
    grads = {}
    for sg in (False, True):
        out = losses(model, SIGReg(), emb, {"loss": cfg["loss"] | {"stop_grad_target": sg}})
        grads[sg] = (out["tf"].item(), torch.autograd.grad(out["tf"], emb)[0])
    assert grads[False][0] == grads[True][0]
    assert grads[False][1][:, -1].abs().sum() > 0 and grads[True][1][:, -1].abs().sum() == 0  # last state: target only
    assert grads[True][1][:, 0].abs().sum() > 0  # first state: read by the predictor


def test_rollout_starts_at_the_first_step_of_its_epoch():
    from h2r.training.world_model import epoch_of, rollout_start

    assert [epoch_of(s, 10) for s in (1, 10, 11, 20, 21)] == [0, 0, 1, 1, 2]
    assert rollout_start({"rollout_from_epoch": 0}, 10) == 1  # from the start
    assert rollout_start({"rollout_from_epoch": 2}, 10) == 21 and epoch_of(21, 10) == 2
    assert rollout_start({"rollout_from": 1050}, 10) == 1050  # runs saved before the key: a step
    with pytest.raises(ValueError):
        rollout_start({"rollout_from_epoch": 1, "rollout_from": 1050}, 10)


@pytest.fixture
def tiny_wm_cfg(tmp_path):
    """configs/training/world_model.yaml on 28 px clips of random JPEGs, a model of a few thousand parameters"""
    import io

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    from h2r.cli import load

    rng = np.random.default_rng(0)

    def jpeg():
        buf = io.BytesIO()
        Image.fromarray(rng.integers(0, 255, (28, 28, 3), dtype=np.uint8)).save(buf, format="JPEG")
        return buf.getvalue()

    (tmp_path / "data" / "frames").mkdir(parents=True)
    for split, n in [("train", 6), ("val", 3)]:
        rows = [{"video": f"{split}{i}.mp4", "frames": [jpeg() for _ in range(6)]} for i in range(n)]
        pq.write_table(pa.Table.from_pylist(rows), tmp_path / "data" / "frames" / f"{split}-00000.parquet")
    cfg = load("configs/training/world_model.yaml")
    D, Z = 16, 8
    cfg["model"] = {
        "embed_dim": D, "latent_action_dim": Z, "num_patches": 1, "residual": False, "history_size": 2,
        "encoder": {"type": "vit", "image_size": 28, "patch_size": 14, "dim": 16, "depth": 1, "heads": 2},
        "projector": {"hidden_dim": 32, "output_dim": D},
        "predictor": {"input_dim": D, "hidden_dim": D, "cond_dim": Z, "depth": 1, "heads": 2, "dim_head": 8, "mlp_dim": 32},
        "pred_proj": {"input_dim": D, "hidden_dim": 32, "output_dim": D},
        "idm": {"dim": D, "z_dim": Z, "depth": 1, "heads": 2},
    }
    cfg["data"] |= {"processed_dir": str(tmp_path / "data"), "clip_len": 4}
    cfg["loss"] |= {"rollout_steps": 2, "rollout_from_epoch": 1, "rollout_window": 2}
    cfg["probe"] |= {"hands": None, "videos": 0}
    cfg |= {"output_dir": str(tmp_path / "out"), "precision": "fp32", "batch_size": 4, "num_workers": 0,
            "iterations": 6, "log_every": 2, "eval_every": 2, "save_every": 2}
    return cfg


def test_train_world_model_and_resume(tiny_wm_cfg):
    """train writes the metrics and checkpoints; a resumed run continues from last.pt and drops the lines logged after it"""
    import json
    from pathlib import Path

    import yaml

    from h2r.training.world_model import train

    train(tiny_wm_cfg | {"iterations": 4})
    run = next(Path(tiny_wm_cfg["output_dir"]).iterdir())
    assert {"best.pt", "last.pt", "step-2.pt", "step-4.pt", "metrics.jsonl", "config.yaml"} <= {p.name for p in run.iterdir()}
    assert [json.loads(line)["step"] for line in open(run / "metrics.jsonl")] == [2, 2, 4, 4]  # train and val lines

    with open(run / "metrics.jsonl", "a") as f:  # as if the run had logged step 5 then crashed before saving
        f.write(json.dumps({"step": 5, "loss/total": 0.0}) + "\n")
    saved = yaml.safe_load(open(run / "config.yaml"))
    yaml.safe_dump(saved | {"iterations": 6}, open(run / "config.yaml", "w"))
    train({"resume": str(run)})
    lines = [json.loads(line) for line in open(run / "metrics.jsonl")]
    assert [line["step"] for line in lines] == [2, 2, 4, 4, 6, 6]
    assert torch.load(run / "last.pt", weights_only=False)["step"] == 6
    assert any("loss/rollout_2" in line for line in lines if line["step"] == 6)  # k-step loss from epoch 1
