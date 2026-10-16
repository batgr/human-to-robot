import json

import numpy as np
import pytest
import torch

from h2r.data.dataset import select
from h2r.eval.counterfactual import scores
from h2r.eval.pick import wilson
from h2r.eval.planning import displacement_error
from h2r.eval.transfer import interval, load_successes, savings
from h2r.training.controller import select_transitions


def test_budgets_are_shared_and_nested():
    """A budget keeps the same episodes whatever the order of the list, and a smaller budget is part of a larger one"""
    videos = [f"{r}/Lift/scripted/episode_{i:06d}/agentview.mp4" for r in ("Panda", "Sawyer") for i in range(20)]
    assert select(videos, ["Sawyer/*"]) == list(range(20, 40))
    small, large = (set(videos[i] for i in select(videos, ["Panda/*"], n, seed=0)) for n in (4, 10))
    assert len(small) == 4 and small < large and all(v.startswith("Panda/") for v in large)
    reordered = videos[::-1]
    assert {reordered[i] for i in select(reordered, ["Panda/*"], 4, seed=0)} == small
    per_transition = [v for v in videos for _ in range(3)]  # the controller: 3 transitions per episode
    data = {"s": torch.arange(120.0), "a": torch.arange(120.0), "mu": torch.arange(120.0), "video": per_transition}
    sub = select_transitions(data, ["Panda/*"], max_videos=4, seed=0)
    assert len(sub["s"]) == 12 and {per_transition[int(i)] for i in sub["s"]} == small


def test_paired_interval_and_savings():
    rng = np.random.default_rng(0)
    a = (rng.random((2, 100)) < 0.3).astype(float)
    point, (lo, hi) = interval(a, a, draws=2000)
    assert point == 0 and lo <= 0 <= hi
    better = np.maximum(a, (rng.random((2, 100)) < 0.4))  # succeeds wherever a does, and more
    point, (lo, hi) = interval(a, better, draws=2000)
    assert point > 0 and lo > 0
    mean, (lo, hi) = interval(a, draws=2000)
    assert lo < mean < hi and abs(mean - a.mean()) < 1e-12
    ref = {10: 0.1, 50: 0.4, 250: 0.7}
    assert savings(ref, 0.3) == (10, 50) and savings(ref, 0.05) == (None, 10) and savings(ref, 0.9) == (250, None)
    lo, hi = wilson(8, 10)
    assert 0.4 < lo < 0.8 < hi < 1 and wilson(0, 0) == (0.0, 0.0)


def test_runs_must_share_the_initial_states(tmp_path):
    def result(name, episodes, wins):
        rows = [{"episode": e, "planner": {"success": w}} for e, w in zip(episodes, wins)]
        json.dump({"episodes": rows}, open(tmp_path / name, "w"))
        return str(tmp_path / name)

    runs = [{"arm": "A", "episodes": 10, "seed": s, "path": result(f"a{s}", ["e1", "e2"], [s, 0])} for s in (0, 1)]
    runs.append({"arm": "C", "episodes": 10, "seed": 0, "path": result("c", ["e1", "e2"], [1, 1])})
    success, states = load_successes(runs, "planner")
    assert states == ["e1", "e2"] and success["A", 10].tolist() == [[0, 0], [1, 0]]
    runs.append({"arm": "C", "episodes": 50, "seed": 0, "path": result("d", ["e1", "e3"], [1, 1])})
    with pytest.raises(ValueError):
        load_successes(runs, "planner")


def test_counterfactual_scores():
    real = torch.randn(8, 32)
    perfect = scores(real.clone(), real)
    assert perfect["identification"] == 1 and abs(perfect["direction"] - 1) < 1e-5 and abs(perfect["spread"] - 1) < 1e-5
    blind = scores(real.mean(0, keepdim=True).expand(8, 32) + 1e-3 * torch.randn(8, 32), real)
    assert blind["spread"] < 0.01


def test_displacement_error_compares_summed_translations():
    recorded = torch.tensor([[0.01, 0.0, 0.0, 0.3, 0, 0, 0.2], [0.01, 0.02, 0.0, 0, 0, 0, 0]])
    other_path = torch.tensor([[0.0, 0.02, 0.0, 0, 0, 0, 0], [0.02, 0.0, 0.0, 0, 0, 0, 0]])
    assert abs(displacement_error(other_path, recorded)) < 1e-5  # same end point, rotation and gripper ignored
    assert abs(displacement_error(torch.zeros_like(recorded), recorded) - 4.0) < 1e-4  # 2 + 2 cm
