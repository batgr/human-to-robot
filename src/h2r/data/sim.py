"""Collect robosuite episodes (frames + real actions) for the controller, as frame shards with extra columns.

Two policies: noisy scripted waypoints (reach, grasp, lift, or random targets) and random actions.
Each action is held for frameskip control steps, so one action per model step (20 Hz / 5 = 4 Hz).
"""

import io
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from h2r.data.prepare import split_of


def scripted(rng, obs, task, frameskip):
    """Generator of actions toward noisy waypoints; receives the new observation after each action"""
    low, high = np.array(task["low"]), np.array(task["high"])
    target = obs[task["object"]]
    if rng.random() < 2 / 3:  # reach, grasp, lift
        off = np.r_[rng.normal(0, 0.02, 2), 0]
        wps = [(target + off + [0, 0, 0.08], -1), (target + off, -1), (target + off, 1), (target + [0, 0, 0.12], 1)]
    else:  # random targets in the workspace
        wps = [(rng.uniform(low, high), rng.choice([-1, 1])) for _ in range(5)]
    w, hold, noise = 0, 0, np.zeros(3)
    while True:
        goal, grip = wps[w]
        delta = np.clip(goal, low, high) - obs["robot0_eef_pos"]
        noise = 0.7 * noise + rng.normal(0, 0.3, 3)
        a = np.zeros(7)
        a[:3] = np.clip(delta / (frameskip * 0.05) * rng.uniform(0.5, 1.0) + noise, -1, 1)
        a[6] = grip
        obs = yield a
        hold += 1
        if np.linalg.norm(delta) < 0.02 or hold > 8:
            w, hold = min(w + 1, len(wps) - 1), 0


def random(rng, obs, task, frameskip):
    """Uniform random translation and gripper; pushes outside the workspace are cancelled"""
    low, high = np.array(task["low"]), np.array(task["high"])
    while True:
        a = np.zeros(7)
        a[:3] = rng.uniform(-1, 1, 3)
        a[6] = rng.choice([-1, 1])
        eef = obs["robot0_eef_pos"]
        a[:3][((eef <= low) & (a[:3] < 0)) | ((eef >= high) & (a[:3] > 0))] = 0  # keep the gripper in view
        obs = yield a


POLICIES = {"scripted": scripted, "random": random}


def jpeg(image):
    buf = io.BytesIO()
    Image.fromarray(image[::-1]).save(buf, format="JPEG", quality=95)  # robosuite renders upside down
    return buf.getvalue()


ENVS = {}  # one environment per task in each worker process


def episode(args):
    """One episode: steps + 1 frames, steps actions, end effector, object and simulator states"""
    cfg, name, policy, seed = args
    import robosuite as suite  # in the worker: each process has its own renderer

    task = cfg["tasks"][name]
    if name not in ENVS:
        ENVS[name] = suite.make(name, **cfg["env"])
    env = ENVS[name]
    np.random.seed(seed)  # robosuite places the objects with the global generator
    rng = np.random.default_rng(seed)
    obs = env.reset()
    act = POLICIES[policy](rng, obs, task, cfg["frameskip"])
    a = next(act)
    row = {"frames": [], "actions": [], "eef": [], "object": [], "states": []}
    for t in range(cfg["steps"] + 1):
        row["frames"].append(jpeg(obs[f"{cfg['env']['camera_names']}_image"]))
        row["eef"].append(obs["robot0_eef_pos"].tolist())
        row["object"].append(obs[task["object"]].tolist())
        row["states"].append(env.sim.get_state().flatten().tolist())
        if t == cfg["steps"]:
            break
        for _ in range(cfg["frameskip"]):
            obs, *_ = env.step(a)
        row["actions"].append(a.tolist())
        a = act.send(obs)
    return {"video": f"{name}/{policy}/episode_{seed:06d}", "task": name, "policy": policy} | row


def collect(cfg):
    out = Path(cfg["processed_dir"])
    (out / "frames").mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg["config_path"], out / "config.yaml")
    jobs = [(cfg, name, policy, seed) for name in cfg["tasks"] for policy in cfg["policies"]
            for seed in range(cfg["episodes"])]
    buffers, shards, manifest = {}, {}, []

    def flush(split):
        rows = buffers.pop(split)
        name = f"{split}-{shards.get(split, 0):05d}.parquet"
        shards[split] = shards.get(split, 0) + 1
        pq.write_table(pa.Table.from_pylist(rows), out / "frames" / name)
        manifest.extend({"video": r["video"], "split": split, "n_frames": len(r["frames"]), "shard": name}
                        for r in rows)

    with ProcessPoolExecutor(os.cpu_count()) as pool:
        for i, row in enumerate(pool.map(episode, jobs, chunksize=4)):
            split = split_of(row["video"], cfg["split"])
            buffers.setdefault(split, []).append(row)
            if len(buffers[split]) == cfg["shard_size"]:
                flush(split)
            if (i + 1) % 100 == 0:
                print(f"{i + 1} / {len(jobs)}", flush=True)
    for split in list(buffers):
        flush(split)

    pq.write_table(pa.Table.from_pylist(manifest), out / "manifest.parquet")
    for split in ("train", "val", "test"):
        kept = [m for m in manifest if m["split"] == split]
        print(f"{split}: {len(kept)} episodes, {sum(m['n_frames'] for m in kept)} frames")
