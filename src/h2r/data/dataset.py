import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset


def decode(jpegs):
    """JPEG bytes -> (T, 3, H, W) uint8 frames"""
    frames = np.stack([np.asarray(Image.open(io.BytesIO(j))) for j in jpegs])
    return torch.from_numpy(frames).permute(0, 3, 1, 2)


def shards(root, split):
    return sorted(Path(root, "frames").glob(f"{split}-*.parquet"))


def download(repo, root):
    """Frame shards of a Hub dataset into root, once (the raw videos and simulator states stay on the Hub)"""
    if not shards(root, "train"):
        from huggingface_hub import snapshot_download

        snapshot_download(repo, repo_type="dataset", local_dir=root,
                          allow_patterns=["frames/*", "manifest.parquet", "config.yaml"])


def read_episodes(root, split):
    """Videos of a split that have actions: [{"video", "frames": JPEG bytes, "actions": (T - 1, act_dim)}], and
    "states": (T, state_dim) the absolute state at each frame when the shards have it"""
    rows = []
    for p in shards(root, split):
        columns = [c for c in ["video", "frames", "actions", "states"] if c in pq.read_schema(p).names]
        rows += pq.read_table(p, columns=columns).to_pylist()
    return [r for r in rows if r["actions"] is not None]


def windows(episodes, horizon):
    """Non-overlapping windows of horizon + 1 states of every episode, as many as fit. episodes: {"s": states of each
    episode, (T_e, ...), the others per frame (T_e, ...) e.g. "x" or per transition (T_e - 1, ...) e.g. "a", "mu"}:
    lists, the episodes may differ in length (DROID), or tensors (E, T, ...) -> per frame (W, horizon + 1, ...), per
    transition (W, horizon, ...).
    Ordered by start time then episode: neighbouring windows come from different episodes (ControlledRollout takes
    another episode's actions from the neighbour)"""
    out = {k: [] for k in episodes}
    for t in range(0, max(len(s) for s in episodes["s"]) - horizon, horizon):
        for e, s in enumerate(episodes["s"]):
            if t + horizon < len(s):
                for k, v in episodes.items():
                    out[k].append(v[e][t : t + horizon + (len(v[e]) == len(s))])
    return {k: torch.stack(v) for k, v in out.items()}


def all_actions(root, split):
    """(n, act_dim) every action of a split, e.g. to standardize them"""
    rows = [a for p in shards(root, split) for a in pq.read_table(p, columns=["actions"]).column(0).to_pylist() if a]
    return torch.tensor([x for a in rows for x in a])


class ClipDataset(Dataset):
    """Clips of clip_len consecutive frames of every video; actions = True: with the clip_len - 1 real actions between
    them, from the videos that have actions (the others are left out)"""

    def __init__(self, root, split, clip_len=5, actions=False):
        columns = ["frames", "actions"] if actions else ["frames"]
        table = pa.concat_tables(
            pq.read_table(p, columns=columns, memory_map=True)
            for p in shards(root, split)
        )
        self.frames = table.column("frames").combine_chunks()
        self.actions = table.column("actions").combine_chunks() if actions else None
        self.clip_len = clip_len
        lengths = self.frames.value_lengths().to_numpy()
        if actions:  # no clip from a video without actions
            lengths = np.where(self.actions.is_null().to_numpy(zero_copy_only=False), 0, lengths)

        self.ends = np.cumsum(np.maximum(lengths - clip_len + 1, 0))

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, i):
        """Returns (clip_len, 3, H, W) uint8, and with actions (clip_len - 1, act_dim) float32"""
        v = int(np.searchsorted(self.ends, i, side="right"))
        s = i - (int(self.ends[v - 1]) if v else 0)
        frames = decode(j.as_py() for j in self.frames[v].values[s : s + self.clip_len])
        if self.actions is None:
            return frames
        return frames, torch.tensor(self.actions[v].values[s : s + self.clip_len - 1].to_pylist())
