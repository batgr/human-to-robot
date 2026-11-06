import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from PIL import Image
from test_jepa import small_model

from h2r.data.dataset import windows
from h2r.metrics.control import ControlledRollout
from h2r.models.controller import Controller


def test_rollout_with_the_true_latent_actions_matches_the_idm():
    """A controller that returns the IDM's mu (passed as the action) reproduces the IDM's rollout: ratio 1"""
    model, D = small_model()
    H = 4
    emb, mu = torch.randn(3, H + 1, D), torch.randn(3, H, D)
    control = ControlledRollout()
    control.update(model, lambda a, s: a, emb, mu, actions=mu)
    out = control.compute()
    for h in range(1, H + 1):
        assert out[f"error_h{h}"] == pytest.approx(out[f"error_idm_h{h}"], rel=1e-5)
        assert out[f"error_ratio_h{h}"] == pytest.approx(1, rel=1e-5)


def test_windows_cut_episodes_without_overlap():
    E, T, D, A, Z = 2, 9, 3, 2, 4
    data = {"s": torch.arange(E * T * D).view(E, T, D), "mu": torch.randn(E, T - 1, Z), "a": torch.randn(E, T - 1, A)}
    w = windows(data, horizon=3)  # starts 0, 3 (6 + 3 = 9 states would need T = 10)
    assert w["s"].shape == (2 * E, 4, D) and w["mu"].shape == (2 * E, 3, Z) and w["a"].shape == (2 * E, 3, A)
    assert torch.equal(w["s"][E], data["s"][0, 3:7]) and torch.equal(w["a"][E], data["a"][0, 3:6])


def test_windows_of_episodes_of_different_lengths():
    s = [torch.arange(9)[:, None].float(), torch.arange(5)[:, None].float() + 100]
    a = [torch.zeros(8, 2), torch.ones(4, 2)]
    w = windows({"s": s, "a": a}, horizon=3)  # starts 0, 3 in the first, 0 in the second
    assert w["s"].shape == (3, 4, 1) and w["a"].shape == (3, 3, 2)
    assert w["s"][:, 0, 0].tolist() == [0, 100, 3]  # by start time, then episode


def jpeg(rng):
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (28, 28, 3), dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def tiny_runs(tmp_path):
    """A tiny world model run (config.yaml + best.pt) and episodes with actions in every split"""
    from h2r.models.jepa import build_model

    D, Z = 16, 8
    model_cfg = {
        "num_patches": 1, "history_size": 2,
        "encoder": {"type": "vit", "image_size": 28, "patch_size": 14, "dim": 16, "depth": 1, "heads": 2},
        "projector": {"hidden_dim": 32, "output_dim": D},
        "predictor": {"input_dim": D, "hidden_dim": D, "cond_dim": Z, "depth": 1, "heads": 2, "dim_head": 8, "mlp_dim": 32},
        "pred_proj": {"input_dim": D, "hidden_dim": 32, "output_dim": D},
        "idm": {"dim": D, "z_dim": Z, "depth": 1, "heads": 2},
    }
    wm = tmp_path / "wm-tiny"
    wm.mkdir()
    yaml.safe_dump({"model": model_cfg}, open(wm / "config.yaml", "w"))
    model = build_model(model_cfg)
    torch.nn.init.normal_(model.idm.head.weight, std=0.1)  # mu = 0 at initialization, as if trained
    torch.save({"model": model.state_dict()}, wm / "best.pt")

    rng = np.random.default_rng(0)
    schema = pa.schema([("video", pa.string()), ("frames", pa.list_(pa.binary())),
                        ("actions", pa.list_(pa.list_(pa.float32()))), ("states", pa.list_(pa.list_(pa.float32())))])
    (tmp_path / "sim" / "frames").mkdir(parents=True)
    for split, n in [("train", 6), ("val", 2), ("test", 2)]:
        lengths = [9 + i for i in range(n)]  # episodes of different lengths, as DROID's
        rows = [{"video": f"droid/episode_{split}{i}/exterior_1_left.mp4", "frames": [jpeg(rng) for _ in range(T)],
                 "actions": rng.normal(size=(T - 1, 7)).astype(np.float32).tolist(),
                 "states": rng.normal(size=(T, 7)).astype(np.float32).tolist()} for i, T in enumerate(lengths)]
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp_path / "sim" / "frames" / f"{split}-00000.parquet")
    return tmp_path, wm


def test_train_controller_end_to_end(tiny_runs):
    """train writes best.pt and the val and test metrics; load_controller gives back the frozen pair"""
    from h2r.cli import load
    from h2r.models.controller import load_controller
    from h2r.training.controller import train

    tmp, wm = tiny_runs
    cfg = load("configs/training/controller.yaml")
    cfg |= {"world_model": str(wm), "output_dir": str(tmp / "out"), "precision": "fp32", "horizon": 3,
            "iterations": 4, "batch_size": 8, "log_every": 2, "eval_every": 2}
    cfg["model"]["mlp_dim"] = 32
    cfg["data"] |= {"root": str(tmp / "sim"), "cache": str(tmp / "cache")}
    train(cfg)

    run = next((tmp / "out").iterdir())
    lines = [json.loads(line) for line in open(run / "metrics.jsonl")]
    test = next(line for line in lines if "test_sim/loss" in line)  # test_<name of the data root>, one per test set
    assert any("val/loss" in line for line in lines)
    assert {"test_sim/loss", "test_sim_rollout/error_h3", "test_sim_rollout/error_idm_h3", "test_sim_rollout/error_ratio_h3"} <= test.keys()
    assert (tmp / "cache" / "wm-tiny" / "sim-train.pt").exists()
    model, ctrl = load_controller(run, torch.device("cpu"))
    assert not any(p.requires_grad for p in ctrl.parameters())
