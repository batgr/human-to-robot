import io

import numpy as np
import torch
from PIL import Image

from h2r.data.dataset import decode


def jpeg(value):
    buf = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), value, np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def test_decode():
    frames = decode([jpeg(0), jpeg(200)])
    assert frames.shape == (2, 3, 8, 8) and frames.dtype == torch.uint8 and abs(int(frames[1].float().mean()) - 200) < 3
