import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from h2r.data.dataset import ClipDataset, allocate, decode, fetch, select


def jpeg(value):
    buf = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), value, np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def test_decode():
    frames = decode([jpeg(0), jpeg(200)])
    assert frames.shape == (2, 3, 8, 8) and frames.dtype == torch.uint8 and abs(int(frames[1].float().mean()) - 200) < 3


def fake_hub(tmp_path, monkeypatch, subsets):
    """A Hub dataset as a local folder: subsets {name: (videos per split, frames per video, fps)}, shards of 4 videos"""
    import shutil

    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    hub = tmp_path / "hub"
    for s, (n, length, fps) in subsets.items():
        (hub / s / "frames").mkdir(parents=True)
        (hub / s / "config.yaml").write_text(yaml.safe_dump({"fps": fps, "image_size": 8}))
        manifest = []
        for split in ("train", "val"):
            names = [f"{s}-{split}-{i}.mp4" for i in range(n)]
            for k in range(0, n, 4):
                shard = f"{split}-{k // 4:05d}.parquet"
                rows = [{"video": v, "frames": [jpeg(0)] * length} for v in names[k : k + 4]]
                pq.write_table(pa.Table.from_pylist(rows), hub / s / "frames" / shard)
                manifest += [{"video": r["video"], "split": split, "n_frames": length, "shard": shard} for r in rows]
        pq.write_table(pa.Table.from_pylist(manifest), hub / s / "manifest.parquet")
        pq.write_table(pa.table({"video": [r["video"] for r in manifest], "label": [0] * len(manifest)}),
                       hub / s / "labels.parquet")
    files = [str(p.relative_to(hub)) for p in hub.rglob("*") if p.is_file()]
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", lambda self, repo, repo_type: files)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, f, repo_type: str(hub / f))

    def snapshot(repo, repo_type, local_dir, allow_patterns):
        for f in allow_patterns:
            (Path(local_dir) / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(hub / f, Path(local_dir) / f)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)


def test_allocate_shares_equally_and_redistributes():
    take = allocate({"a": 1.0, "b": 10.0, "c": 10.0}, 9.0)
    assert take == {"a": 1.0, "b": 4.0, "c": 4.0}
    assert allocate({"a": 1.0, "b": 2.0}, 100.0) == {"a": 1.0, "b": 2.0}


def test_select_nested_and_per_split():
    manifest = [{"video": f"{sp}{i}", "split": sp, "n_frames": 10, "shard": f"{sp}-{i // 4}"}
                for sp, n in [("train", 40), ("val", 8)] for i in range(n)]
    small, large = select(manifest, 0.25, 0, "a"), select(manifest, 0.5, 0, "a")
    assert sum(r["split"] == "train" for r in small) == 10 and sum(r["split"] == "val" for r in small) == 2
    assert {r["video"] for r in small} <= {r["video"] for r in large}
    assert len({r["shard"] for r in small if r["split"] == "train"}) == 3  # 2 whole shards + 2 videos of a third
    assert len(select(manifest, 1.0, 0)) == 48


def test_fetch_balanced_mix_reads_with_clip_dataset(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    # 3600 frames at 1 fps = 1 h per video; big: 12 train + 12 val videos, small: 4 + 4
    fake_hub(tmp_path, monkeypatch, {"big": (12, 3600, 1), "small": (4, 3600, 1)})
    fetch("me/h2r-human", tmp_path / "out", hours=16)  # 8 h each: all of small, a third of big
    m = pq.read_table(tmp_path / "out" / "manifest.parquet").to_pylist()
    hours = {s: sum(r["subset"] == s for r in m) for s in ("big", "small")}
    assert hours == {"big": 8, "small": 8}
    assert sum(r["subset"] == "big" and r["split"] == "train" for r in m) == 4  # each split the same share
    ds = ClipDataset(tmp_path / "out", "train", clip_len=5)
    assert len(ds) == 8 * (3600 - 4) and ds[0].shape == (5, 3, 8, 8)
    labels = pq.read_table(tmp_path / "out" / "labels.parquet")
    assert sorted(labels["video"].to_pylist()) == sorted(r["video"] for r in m)
    assert not (tmp_path / "out" / ".download").exists()

    fetch("me/h2r-human", tmp_path / "out", hours=16)  # same request: nothing to do
    with pytest.raises(ValueError, match="another selection"):
        fetch("me/h2r-human", tmp_path / "out", hours=4)
    fetch("me/h2r-human", tmp_path / "only", subsets="small", hours=2)
    assert {r["subset"] for r in pq.read_table(tmp_path / "only" / "manifest.parquet").to_pylist()} == {"small"}


def test_fetch_refuses_mixed_frame_rates(tmp_path, monkeypatch):
    fake_hub(tmp_path, monkeypatch, {"a": (4, 10, 4), "b": (4, 10, 8)})
    with pytest.raises(ValueError, match="different frames"):
        fetch("me/h2r-human", tmp_path / "out")


def write_shards(root, lengths_per_shard):
    """frames/train-NNNNN.parquet, video k frame t = a JPEG of value (10 * k + t) % 250"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "frames").mkdir(parents=True)
    k = 0
    for n, lengths in enumerate(lengths_per_shard):
        rows = []
        for length in lengths:
            rows.append({"video": f"v{k}", "frames": [jpeg((10 * k + 3 * t) % 250) for t in range(length)]})
            k += 1
        pq.write_table(pa.Table.from_pylist(rows), root / "frames" / f"train-{n:05d}.parquet")


def test_clip_dataset_stride_and_order(tmp_path):
    write_shards(tmp_path, [[7, 3], [12]])  # video 1 too short for a clip of 5
    every = ClipDataset(tmp_path, "train", clip_len=5)
    assert len(every) == 3 + 0 + 8
    strided = ClipDataset(tmp_path, "train", clip_len=5, stride=3)
    assert len(strided) == 1 + 0 + 3  # starts 0 | - | 0, 3, 6
    level = lambda clip: [round(float(f.float().mean()) / 3) for f in clip]  # JPEG values back to t
    assert level(strided[2]) == level(every[3 + 3])  # video 2, start 3
    assert (tmp_path / "frames" / "train-00000.arrow").exists()


def test_clip_dataset_pickles_without_frames(tmp_path):
    import pickle

    write_shards(tmp_path, [[6]])
    ds = ClipDataset(tmp_path, "train", clip_len=5)
    copy = pickle.loads(pickle.dumps(ds))  # what a spawned DataLoader worker receives
    assert copy.chunks is None and len(pickle.dumps(ds)) < 2000
    assert torch.equal(copy[1], ds[1])
