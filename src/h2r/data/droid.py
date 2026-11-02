"""Download DROID episodes (LeRobot v3, lerobot/droid_1.0.1) as one folder per episode, ready for extract:
raw_dir/episode_XXXXXX/{camera}.mp4 and state.parquet (timestamp, state = end-effector xyz + Euler xyz + gripper
closedness, the 7D state of V-JEPA 2-AC, from which extract computes the actions)

LeRobot v3 packs many episodes in each mp4 and each data parquet: meta/episodes gives the files and time span of every
episode. A seeded draw of max_files video files bounds the download, then random episodes from them until the budget.
"""

import random
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

STATE = ["observation.state.cartesian_position", "observation.state.gripper_position"]


def cut(video, start, end, out):
    """Copy [start, end] seconds of video to out (re-encoded, so the cut is frame accurate)"""
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start}", "-i", str(video), "-t", f"{end - start}",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(out)]
    return subprocess.run(cmd).returncode == 0


def episodes(repo, camera):
    """Every episode: {index, video, start, end, data} (data: the parquet of its states)"""
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith("meta/episodes/")]
    key = f"videos/{camera}"
    cols = ["episode_index", "data/chunk_index", "data/file_index", f"{key}/chunk_index", f"{key}/file_index",
            f"{key}/from_timestamp", f"{key}/to_timestamp"]
    rows = []
    for f in sorted(files):
        for r in pq.read_table(hf_hub_download(repo, f, repo_type="dataset"), columns=cols).to_pylist():
            rows.append({
                "index": r["episode_index"],
                "video": f"{key}/chunk-{r[f'{key}/chunk_index']:03d}/file-{r[f'{key}/file_index']:03d}.mp4",
                "start": r[f"{key}/from_timestamp"],
                "end": r[f"{key}/to_timestamp"],
                "data": f"data/chunk-{r['data/chunk_index']:03d}/file-{r['data/file_index']:03d}.parquet",
            })
    return rows


def write_states(repo, data_file, chosen, raw):
    """state.parquet of each chosen episode of one data file, timestamps from 0 at the episode's first frame"""
    path = hf_hub_download(repo, data_file, repo_type="dataset")
    table = pq.read_table(path, columns=["episode_index", "timestamp", *STATE])
    index = table["episode_index"].to_numpy()
    for e in chosen:
        rows = table.filter(pa.array(index == e["index"]))
        times = rows["timestamp"].to_numpy().astype(np.float64)
        pose = np.asarray(rows[STATE[0]].to_pylist(), dtype=np.float64)
        grip = np.asarray(rows[STATE[1]].to_pylist(), dtype=np.float64).reshape(len(pose), -1)
        state = np.concatenate([pose, grip], axis=1)
        out = raw / f"episode_{e['index']:06d}"
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"timestamp": times - times[0], "state": state.tolist()}), out / "state.parquet")


def download(cfg):
    d = cfg["droid"]
    raw = Path(cfg["raw_dir"])
    rng = random.Random(d["seed"])
    rows = [r for r in episodes(d["repo"], d["camera"]) if r["end"] - r["start"] >= d["min_seconds"]]
    files = sorted({r["video"] for r in rows})
    keep = set(rng.sample(files, min(d["max_files"], len(files))))  # bounds the download
    rows = [r for r in rows if r["video"] in keep]
    rng.shuffle(rows)

    chosen, seconds = [], 0.0
    for r in rows:  # random episodes until the time budget
        if seconds >= d["minutes"] * 60:
            break
        chosen.append(r)
        seconds += r["end"] - r["start"]
    print(f"{len(chosen)} episodes, {seconds / 60:.0f} min, from {len(keep)} of {len(files)} video files", flush=True)

    by_data = defaultdict(list)
    for r in chosen:
        by_data[r["data"]].append(r)
    for data_file, eps in sorted(by_data.items()):
        write_states(d["repo"], data_file, eps, raw)

    by_video = defaultdict(list)
    for r in chosen:
        by_video[r["video"]].append(r)
    done = 0
    for i, (path, eps) in enumerate(sorted(by_video.items())):
        with tempfile.TemporaryDirectory() as tmp:  # the packed file is deleted after cutting
            video = hf_hub_download(d["repo"], path, repo_type="dataset", local_dir=tmp)
            with ThreadPoolExecutor(8) as pool:
                done += sum(pool.map(lambda e: cut(video, e["start"], e["end"],
                                                   raw / f"episode_{e['index']:06d}" / f"{d['camera']}.mp4"), eps))
        print(f"video files {i + 1} / {len(by_video)}: {done} episodes cut", flush=True)
    print(f"{done} / {len(chosen)} episodes in {raw}")
