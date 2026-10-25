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
