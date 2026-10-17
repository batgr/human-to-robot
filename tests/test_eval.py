import torch

from h2r.data.dataset import select
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
