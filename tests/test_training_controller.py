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
from h2r.metrics.control import (
    ControlledRollout,
    PlanningError,
    action_classes,
    action_probe,
    classify,
    plan,
    regimes,
)
from h2r.models.controller import Controller


def test_rollout_with_the_true_latent_actions_keeps_the_oracle_skill():
    """A controller that returns the IDM's mu (passed as the action) reproduces the oracle: lost = 0;
    the mean baseline is the rollout of the constant mean latent action"""
    model, D = small_model()
    H = 4
    emb, mu = torch.randn(3, H + 1, D), torch.randn(3, H, D)
    mean_z = torch.randn(D)
    control = ControlledRollout(mean_z)
    control.update(model, lambda a, s: a, emb, mu, actions=mu)
    out = control.compute()
    for h in range(1, H + 1):
        assert out[f"skill_h{h}"] == pytest.approx(out[f"skill_idm_h{h}"], abs=1e-5)
        assert out[f"lost_h{h}"] == pytest.approx(0, abs=1e-5)
    mean_pred = model.rollout(emb[:, :1], mu[:, :0], mean_z.expand(3, H, D)[:, None])[:, 0]
    err = (mean_pred - emb[:, 1:]).pow(2).mean((0, 2))
    pers = (emb[:, :1] - emb[:, 1:]).pow(2).mean((0, 2))
    assert out["skill_mean_h2"] == pytest.approx(float(1 - err[1] / pers[1]), abs=1e-5)


def test_action_probe_reads_linear_actions_and_not_noise():
    torch.manual_seed(0)
    mu, w = torch.randn(2000, 16), torch.randn(16, 7)
    perfect = action_probe(mu[:1000], mu[:1000] @ w, mu[1000:], mu[1000:] @ w)
    noise = action_probe(mu[:1000], torch.randn(1000, 7), mu[1000:], torch.randn(1000, 7))
    assert perfect["r2"] > 0.999 and abs(noise["r2"]) < 0.05 and "r2_gripper" in perfect


def test_planning_error_against_planning_no_motion():
    true = torch.randn(5, 3, 7)
    error = PlanningError()
    error.update(true.clone(), true)
    assert error.compute()["dxyz"] == pytest.approx(0)
    error = PlanningError()
    error.update(torch.zeros_like(true), true)
    assert error.compute()["dxyz_ratio"] == pytest.approx(1)


def test_cem_gets_closer_to_the_goal_than_no_action():
    """The goal is reached by known actions through the same controller and world model: the plan's cost is below
    the cost of the mean action"""
    model, D = small_model()
    torch.manual_seed(0)
    ctrl = Controller(dim=D, z_dim=D, act_dim=7, mlp_dim=32).eval()
    torch.nn.init.normal_(ctrl.head.weight, std=0.5)
    start, true = torch.randn(4, D), torch.randn(4, 3, 7)
    cost = lambda acts: (model.rollout(start[:, None], start.new_zeros(4, 0, D), acts[:, None], ctrl)[:, 0, -1]
                         - goal).pow(2).sum(-1)
    goal = model.rollout(start[:, None], start.new_zeros(4, 0, D), true[:, None], ctrl)[:, 0, -1]
    planned = plan(model, ctrl, start[:, None], start.new_zeros(4, 0, D), goal, horizon=3, samples=64, elites=8,
                   iterations=5, generator=torch.Generator().manual_seed(0))
    assert planned.shape == (4, 3, 7)
    assert (cost(planned) < cost(torch.zeros_like(true))).all()
    bounds = torch.tensor([0.1] * 6 + [0.75])
    bounded = plan(model, ctrl, start[:, None], start.new_zeros(4, 0, D), goal, horizon=3, samples=64, elites=8,
                   iterations=2, generator=torch.Generator().manual_seed(0), bounds=bounds)
    assert (bounded.abs() <= bounds + 1e-6).all()


def test_action_probe_leaves_constant_dimensions_out():
    torch.manual_seed(0)
    mu, w = torch.randn(2000, 16), torch.randn(16, 7)
    a = mu @ w
    a[:, 3:6] = 0  # rotations never commanded (sim)
    out = action_probe(mu[:1000], a[:1000], mu[1000:], a[1000:])
    assert out["constant"] == 3 and out["r2_roll"] is None and out["r2"] > 0.999


def test_action_classes_and_classifiers():
    a = torch.tensor([[2.0, 0.1, 0, 0, 0, 0, 0.2], [0, -3, 0.5, 0, 0, 0, -0.3], [0, 0, 0.1, 0, 0, 0, 0]])
    labels = action_classes(a, torch.ones(7))
    assert labels["direction"].tolist() == [0, 3, 4] and labels["gripper"].tolist() == [0, 1, -1]
    torch.manual_seed(0)
    y = torch.randint(6, (600,))
    x = torch.nn.functional.one_hot(y, 6).float() * 3 + 0.1 * torch.randn(600, 6)
    out = classify(x[:400], y[:400], x[400:], y[400:], 6)
    assert out["kNN"][0] > 0.99 and out["linear"][0] > 0.99


def test_windows_cut_episodes_without_overlap():
    E, T, D, A, Z = 2, 9, 3, 2, 4
    data = {"s": torch.arange(E * T * D).view(E, T, D), "mu": torch.randn(E, T - 1, Z), "a": torch.randn(E, T - 1, A)}
    w = windows(data, horizon=3)  # starts 0, 3 (6 + 3 = 9 states would need T = 10)
    assert w["s"].shape == (2 * E, 4, D) and w["mu"].shape == (2 * E, 3, Z) and w["a"].shape == (2 * E, 3, A)
    assert torch.equal(w["s"][E], data["s"][0, 3:7]) and torch.equal(w["a"][E], data["a"][0, 3:6])


def test_regimes_of_windows():
    a = torch.zeros(3, 2, 7)
    a[1, 0, 0] = 0.02  # moves 2 cm at one step
    a[2, 1, 6] = 0.3  # closes the gripper
    assert regimes(a).tolist() == [0, 1, 2]


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
            "iterations": 4, "batch_size": 8, "log_every": 2, "eval_every": 2,
            "planning": {"videos": 2, "clips_per_video": 2, "context": 2, "horizon": 2, "samples": 8, "elites": 2,
                         "iterations": 2, "init_std": 1.0, "bounds": [0.1] * 6 + [0.75]},
            "analysis": {"max_points": 100, "gripper_change": 0.1, "still_xyz": 0.005, "counterfactual_states": 8,
                         "trajectories": 4}}
    cfg["model"]["mlp_dim"] = 32
    cfg["data"] |= {"root": str(tmp / "sim"), "cache": str(tmp / "cache")}
    train(cfg)

    run = next((tmp / "out").iterdir())
    lines = [json.loads(line) for line in open(run / "metrics.jsonl")]
    test = next(line for line in lines if "test/loss" in line)
    assert {"test_rollout/skill_h3", "test_rollout/lost_h3", "test_rollout/error_ratio_h3", "test_rollout/controllability",
            "test_probe/r2_x", "test_planning/dxyz", "test_planning/dxyz_ratio", "test_classes/direction_mu_kNN",
            "test_counterfactual/x_consistency", "test_counterfactual/separability_kNN", "test_state_probe/real_r2",
            "test_state_probe/ctrl_r2_h3", "test_spectra/ds_imagined_effective_rank", "test_planning/clipped_last",
            "test_regimes/free_n"} <= test.keys()
    figures = ["actions", "latent_actions", "counterfactual", "regimes", "state_probe", "spectra", "trajectories", "cem"]
    assert all((run / f"{name}.png").exists() for name in figures)
    assert (tmp / "cache" / "wm-tiny" / "sim-train.pt").exists()
    model, ctrl = load_controller(run, torch.device("cpu"))
    assert not any(p.requires_grad for p in ctrl.parameters())
