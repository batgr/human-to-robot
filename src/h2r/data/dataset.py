"""Clips and episodes read from the frame shards made by h2r.data.prepare"""

import io
import random
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset


def select(videos, include=None, max_videos=None, seed=0):
    """Indices of the videos matching one of the include patterns (fnmatch on the video path, e.g. "Panda/Lift/*"),
    at most max_videos of them (data budget): the first ones of a seeded shuffle of the sorted paths, so that a budget
    keeps the same videos for the world model and the controller, and a smaller budget is part of a larger one"""
    idx = sorted((i for i, v in enumerate(videos) if include is None or any(fnmatch(v, p) for p in include)),
                 key=lambda i: videos[i])
    if max_videos is not None and max_videos < len(idx):
        random.Random(seed).shuffle(idx)
        idx = idx[:max_videos]
    return sorted(idx)


class ClipDataset(Dataset):
    """Every window of clip_len consecutive frames of every selected video of a split"""

    def __init__(self, root, split, clip_len=5, include=None, max_videos=None, seed=0):
        shards = sorted(Path(root, "frames").glob(f"{split}-*.parquet"))
        table = pa.concat_tables(pq.read_table(p, columns=["video", "frames"], memory_map=True) for p in shards)
        keep = select(table.column("video").to_pylist(), include, max_videos, seed)
        if len(keep) < len(table):
            table = table.take(keep)
        self.frames = table.column("frames").combine_chunks()  # arrow buffers: shared by loader workers
        self.clip_len = clip_len
        lengths = self.frames.value_lengths().to_numpy()
        self.index = [(v, s) for v, n in enumerate(lengths) for s in range(n - clip_len + 1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        """Returns (clip_len, 3, H, W) uint8; normalization belongs to the encoder"""
        v, s = self.index[i]
        jpegs = self.frames[v].values[s : s + self.clip_len]
        frames = [np.asarray(Image.open(io.BytesIO(j.as_py()))) for j in jpegs]
        return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)


def read_episodes(root, split, include=None):
    """Videos of a split that have actions (videos without, e.g. human ones, are for the world model only),
    optionally only those matching the include patterns"""
    tables = [
        pq.read_table(p, columns=["video", "frames", "actions"])
        for p in sorted(Path(root, "frames").glob(f"{split}-*.parquet"))
    ]
    rows = [row for t in tables for row in t.to_pylist() if row["actions"] is not None]
    return [rows[i] for i in select([r["video"] for r in rows], include)]
