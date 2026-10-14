"""Clips of consecutive frames read from the frame shards made by h2r.data.prepare"""

import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset


class ClipDataset(Dataset):
    """Every window of clip_len consecutive frames of every video of a split"""

    def __init__(self, root, split, clip_len=5):
        shards = sorted(Path(root, "frames").glob(f"{split}-*.parquet"))
        table = pa.concat_tables(pq.read_table(p, memory_map=True) for p in shards)
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
