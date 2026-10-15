"""Turn a folder of raw videos into frame shards, then push them to the Hugging Face Hub.

Optional actions (actions: true): a state.parquet next to the videos of an episode, with columns timestamp
(seconds from the start of the videos) and state ([x, y, z, roll, pitch, yaw, gripper]: end effector position,
Euler angles in radians, gripper closedness). Every camera of the episode gets the same actions, one per pair of
kept frames, as in V-JEPA 2-AC: position and gripper deltas, and the relative rotation between the two frames.
Videos without a state file get no actions.
"""

import hashlib
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from scipy.spatial.transform import Rotation

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


def states_to_actions(states):
    """states (T, 7) -> actions (T - 1, 7): position delta, relative rotation as Euler angles, gripper delta"""
    rot = Rotation.from_euler("xyz", states[:, 3:6])
    turn = (rot[1:] * rot[:-1].inv()).as_euler("xyz")  # R_t+1 R_t^T, always in [-pi, pi]
    return np.concatenate([np.diff(states[:, :3], axis=0), turn, np.diff(states[:, 6:], axis=0)], axis=1)


def frame_actions(state_path, n_frames, fps):
    """Actions between the n_frames kept frames: state at the nearest timestamp of each frame (frame k at k / fps)"""
    table = pq.read_table(state_path)
    times = np.asarray(table["timestamp"].to_pylist())
    states = np.asarray(table["state"].to_pylist(), dtype=np.float64)
    idx = np.abs(times[None] - np.arange(n_frames)[:, None] / fps).argmin(1)
    return states_to_actions(states[idx]).astype(np.float32).tolist()


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
    columns = [("video", pa.string()), ("frames", pa.list_(pa.binary()))]
    if cfg.get("actions"):
        columns.append(("actions", pa.list_(pa.list_(pa.float32()))))
    schema = pa.schema(columns)
    buffers, shards, manifest = {}, {}, []

    def flush(split):
        rows = buffers.pop(split)
        name = f"{split}-{shards.get(split, 0):05d}.parquet"
        shards[split] = shards.get(split, 0) + 1
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), out / "frames" / name)
        manifest.extend({"video": r["video"], "split": split, "n_frames": len(r["frames"]), "shard": name,
                         "actions": r.get("actions") is not None} for r in rows)

    def work(video):
        frames = extract_frames(raw / video, cfg["fps"], cfg["image_size"])
        state = (raw / video).parent / "state.parquet"
        actions = frame_actions(state, len(frames), cfg["fps"]) if cfg.get("actions") and state.exists() else None
        return video, frames, actions

    with ThreadPoolExecutor(os.cpu_count()) as pool:
        for start in range(0, len(videos), 256):  # chunks keep memory bounded
            for video, frames, actions in pool.map(work, videos[start : start + 256]):
                if len(frames) < cfg["min_frames"]:
                    continue  # too short for one clip, or unreadable
                # split_by folder: the cameras of an episode share its folder, so a split
                key = str(Path(video).parent) if cfg.get("split_by") == "folder" else video
                split = split_of(key, cfg["split"])
                row = {"video": video, "frames": frames} | ({"actions": actions} if cfg.get("actions") else {})
                buffers.setdefault(split, []).append(row)
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
