"""Collect robosuite episodes into raw/, in the same layout as robot videos, for h2r data extract (actions: true).
Back from 1ac8d37^ (batgre/h2r-sim-v3 was collected with it), the policies now in h2r.plan.

raw_dir/<robot>/<task>/<policy>/episode_<seed>/
  <camera>.mp4    one frame per action (control_freq / frameskip fps)
  state.parquet   timestamp + state [x, y, z, roll, pitch, yaw, gripper]: end effector in the robot base frame,
                  Euler angles (xyz) in radians, gripper closedness in [0, 1], as in DROID
  sim.parquet     what only simulation has: commanded OSC actions, MuJoCo states, object and end-effector world
                  positions

Each action is held for frameskip control steps, so one action per model step. Every robot of the config plays the
same seeds: same object placements, so episodes can be compared across embodiments.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from h2r.data.prepare import write_video
from h2r.plan import COLLECTION_POLICIES

ENVS = {}  # one environment per robot and task in each worker process


def episode(args):
    """Plays one episode and writes its folder: steps + 1 frames, steps actions"""
    cfg, robot, name, policy, seed = args
    from h2r.envs.robosuite_env import RobosuiteEnv  # in the worker: each process has its own renderer

    task = cfg["tasks"][name]
    if (robot, name) not in ENVS:
        ENVS[robot, name] = RobosuiteEnv({"env_name": name, "robots": robot, "camera_names": cfg["camera"], **cfg["env"]})
    env = ENVS[robot, name]
    obs = env.reset(seed)
    rng = np.random.default_rng(seed)
    act = COLLECTION_POLICIES[policy](rng, obs, task, cfg["frameskip"])
    a = next(act)
    frames, rows, sim = [], [], []
    for t in range(cfg["steps"] + 1):
        frames.append(env.frame())
        rows.append(env.robot_state(obs))
        sim.append({"command": None, "sim_state": env.sim_state().tolist(),
                    "object": obs[task["object"]].tolist(), "eef_world": obs["robot0_eef_pos"].tolist()})
        if t == cfg["steps"]:
            break
        for _ in range(cfg["frameskip"]):
            obs = env.step(a)
        sim[-1]["command"] = a.tolist()
        a = act.send(obs)

    out = Path(cfg["raw_dir"]) / robot / name / policy / f"episode_{seed:06d}"
    out.mkdir(parents=True, exist_ok=True)
    fps = cfg["env"]["control_freq"] / cfg["frameskip"]
    write_video(frames, out / f"{cfg['camera']}.mp4", fps, cfg["video_crf"])
    times = (np.arange(len(rows)) / fps).tolist()
    pq.write_table(pa.table({"timestamp": times, "state": np.stack(rows).tolist()}), out / "state.parquet")
    pq.write_table(pa.Table.from_pylist([{"timestamp": t} | s for t, s in zip(times, sim)]), out / "sim.parquet")
    return out


def collect(cfg):
    raw = Path(cfg["raw_dir"])
    for robot in cfg["robots"]:
        (raw / robot).mkdir(parents=True, exist_ok=True)
        with open(raw / robot / "config.yaml", "w") as f:  # how the episodes were made, overrides included
            yaml.safe_dump({k: v for k, v in cfg.items() if k != "config_path"}, f, sort_keys=False)
    jobs = [(cfg, robot, name, policy, seed) for robot in cfg["robots"] for name in cfg["collect"]
            for policy in cfg["policies"] for seed in range(cfg["episodes"])]
    with ProcessPoolExecutor(cfg.get("workers") or os.cpu_count()) as pool:
        for i, _ in enumerate(pool.map(episode, jobs, chunksize=4)):
            if (i + 1) % 100 == 0:
                print(f"{i + 1} / {len(jobs)}", flush=True)
    print(f"{len(jobs)} episodes in {raw}, robots {cfg['robots']}, camera {cfg['camera']}")
