"""Turn a folder of raw videos into frame shards, then push them to the Hugging Face Hub.

"""

import hashlib
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
JPEG_END = b"\xff\xd9"


def split_of(video, split):
    """Deterministic split from a hash of the video id: adding videos never moves the others"""
    key = f"{split['seed']}:{video}".encode()
    u = int(hashlib.md5(key).hexdigest(), 16) / 16**32  # uniform in [0, 1)
    if u < split["test"]:
        return "test"
    if u < split["test"] + split["val"]:
        return "val"
    return "train"


def extract_frames(path, fps, size):
    """Resample to fps, resize the short side to size, center crop: list of JPEG bytes"""
    vf = f"fps={fps},scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}"
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf, "-q:v", "3", "-f", "image2pipe", "-c:v", "mjpeg", "-"]
    run = subprocess.run(cmd, capture_output=True)
    if run.returncode:
        return []  # unreadable video: dropped
    out = run.stdout
    frames = out.split(JPEG_END)[:-1]  # 0xFFD9 only appears as end of image
    return [f + JPEG_END for f in frames]


def extract(cfg):
    raw, out = Path(cfg["raw_dir"]), Path(cfg["processed_dir"])
    videos = sorted(
        str(p.relative_to(raw))
        for p in raw.rglob("*")
        if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith(".")  # skip macOS ._ files
    )
    print(f"{len(videos)} videos in {raw}")

    (out / "frames").mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg["config_path"], out / "config.yaml")  # how the frames were made
    schema = pa.schema([("video", pa.string()), ("frames", pa.list_(pa.binary()))])
    buffers, shards, manifest = {}, {}, []

    def flush(split):
        rows = buffers.pop(split)
        name = f"{split}-{shards.get(split, 0):05d}.parquet"
        shards[split] = shards.get(split, 0) + 1
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), out / "frames" / name)
        manifest.extend({"video": r["video"], "split": split, "n_frames": len(r["frames"]), "shard": name} for r in rows)

    def work(video):
        return video, extract_frames(raw / video, cfg["fps"], cfg["image_size"])

    with ThreadPoolExecutor(os.cpu_count()) as pool:
        for start in range(0, len(videos), 256):  # chunks keep memory bounded
            for video, frames in pool.map(work, videos[start : start + 256]):
                if len(frames) < cfg["min_frames"]:
                    continue  # too short for one clip, or unreadable
                split = split_of(video, cfg["split"])
                buffers.setdefault(split, []).append({"video": video, "frames": frames})
                if len(buffers[split]) == cfg["shard_size"]:
                    flush(split)
            print(f"{min(start + 256, len(videos))} / {len(videos)}", flush=True)
    for split in list(buffers):
        flush(split)

    pq.write_table(pa.Table.from_pylist(manifest), out / "manifest.parquet")
    for split in ("train", "val", "test"):
        kept = [m for m in manifest if m["split"] == split]
        print(f"{split}: {len(kept)} videos, {sum(m['n_frames'] for m in kept)} frames")
    print(f"dropped (< {cfg['min_frames']} frames or unreadable): {len(videos) - len(manifest)}")


def push(cfg):
    api = HfApi()
    api.create_repo(cfg["hf_repo"], repo_type="dataset", private=cfg["private"], exist_ok=True)
    api.upload_folder(repo_id=cfg["hf_repo"], folder_path=cfg["processed_dir"], repo_type="dataset")
    visibility = "private" if cfg["private"] else "public"
    print(f"pushed {cfg['processed_dir']} to {cfg['hf_repo']} ({visibility})")
