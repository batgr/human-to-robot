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
    """Videos of a split that have actions: [{"video", "frames": JPEG bytes, "actions": (T - 1, act_dim)}]"""
    rows = [
        row
        for p in shards(root, split)
        for row in pq.read_table(p, columns=["video", "frames", "actions"]).to_pylist()
    ]
    return [r for r in rows if r["actions"] is not None]


def windows(episodes, horizon):
    """Non-overlapping windows of horizon + 1 states of every episode, as many as fit. episodes: {"s": states of each
    episode, (T_e, ...), and per transition (T_e - 1, ...) e.g. "a", "mu"}: lists, the episodes may differ in length
    (DROID), or tensors (E, T, ...) -> s (W, horizon + 1, ...), the others (W, horizon, ...).
    Ordered by start time then episode: neighbouring windows come from different episodes (ControlledRollout takes
    another episode's actions from the neighbour)"""
    out = {k: [] for k in episodes}
    for t in range(0, max(len(s) for s in episodes["s"]) - horizon, horizon):
        for e, s in enumerate(episodes["s"]):
            if t + horizon < len(s):
                for k, v in episodes.items():
                    out[k].append(v[e][t : t + horizon + (k == "s")])
    return {k: torch.stack(v) for k, v in out.items()}


class ClipDataset(Dataset):

    def __init__(self, root, split, clip_len=5):
        table = pa.concat_tables(
            pq.read_table(p, columns=["frames"], memory_map=True)
            for p in shards(root, split)
        )
        self.frames = table.column("frames").combine_chunks()
        self.clip_len = clip_len
        lengths = self.frames.value_lengths().to_numpy()

        self.ends = np.cumsum(np.maximum(lengths - clip_len + 1, 0))

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, i):
        """Returns (clip_len, 3, H, W) uint8"""
        v = int(np.searchsorted(self.ends, i, side="right"))
        s = i - (int(self.ends[v - 1]) if v else 0)
        return decode(j.as_py() for j in self.frames[v].values[s : s + self.clip_len])
