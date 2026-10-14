"""Download robot episodes from LeRobot datasets (v3 format) as one mp4 per episode, ready for extract.

LeRobot v3 packs many episodes in each mp4: meta/episodes gives the file and time span of every episode.
"""

import random
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


def cut(video, start, end, out):
    """Copy [start, end] seconds of video to out (re-encoded, so the cut is frame accurate)"""
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start}", "-i", str(video), "-t", f"{end - start}",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(out)]
    return subprocess.run(cmd).returncode == 0


def episodes(repo, camera):
    """All episodes of a dataset: (index, video file, start, end)"""
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith("meta/episodes/")]
    key = f"videos/{camera}"
    rows = []
    for f in sorted(files):
        table = pq.read_table(hf_hub_download(repo, f, repo_type="dataset"))
        for r in table.select(["episode_index", f"{key}/chunk_index", f"{key}/file_index",
                               f"{key}/from_timestamp", f"{key}/to_timestamp"]).to_pylist():
            path = f"{key}/chunk-{r[f'{key}/chunk_index']:03d}/file-{r[f'{key}/file_index']:03d}.mp4"
            rows.append((r["episode_index"], path, r[f"{key}/from_timestamp"], r[f"{key}/to_timestamp"]))
    return rows


def download(cfg):
    raw = Path(cfg["raw_dir"])
    rng = random.Random(cfg["split"]["seed"])
    for src in cfg["sources"]:
        name = src["repo"].split("/")[-1]
        rows = episodes(src["repo"], src["camera"])
        paths = sorted({r[1] for r in rows})
        keep = set(rng.sample(paths, min(src.get("max_files", len(paths)), len(paths))))  # bounds the download
        rows = [r for r in rows if r[1] in keep]
        rng.shuffle(rows)

        chosen, seconds = [], 0
        for r in rows:  # random episodes until the time budget
            if seconds >= src["minutes"] * 60:
                break
            chosen.append(r)
            seconds += r[3] - r[2]

        by_file = defaultdict(list)
        for r in chosen:
            by_file[r[1]].append(r)
        (raw / name).mkdir(parents=True, exist_ok=True)
        done = 0
        for path, eps in sorted(by_file.items()):
            with tempfile.TemporaryDirectory(dir=raw.parent) as tmp:  # the packed file is deleted after cutting
                video = hf_hub_download(src["repo"], path, repo_type="dataset", local_dir=tmp)
                with ThreadPoolExecutor(8) as pool:
                    ok = pool.map(lambda e: cut(video, e[2], e[3], raw / name / f"episode_{e[0]:06d}.mp4"), eps)
                    done += sum(ok)
        print(f"{name}: {done} episodes, {seconds / 60:.0f} min", flush=True)
