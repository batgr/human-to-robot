"""Collect robosuite episodes into raw/, in the same layout as robot videos, for h2r data extract (actions: true).

raw_dir/<robot>/<task>/<policy>/episode_<seed>/
  <camera>.mp4    one frame per action (control_freq / frameskip fps), one file per camera of the config
  state.parquet   timestamp + state [x, y, z, roll, pitch, yaw, gripper]: end effector in the robot base frame,
                  Euler angles (xyz) in radians, gripper closedness in [0, 1], as in DROID
  sim.parquet     what only simulation has: commanded OSC actions, MuJoCo states (to reset the simulator on any
                  frame), object and end-effector world positions

Two policies: noisy scripted waypoints (reach, grasp, lift, or random targets) and random actions.
Each action is held for frameskip control steps, so one action per model step. Every robot of the config plays the
same seeds: same object placements, so episodes can be compared across embodiments.
"""

import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def scripted(rng, obs, task, frameskip):
    """Generator of actions toward noisy waypoints; receives the new observation after each action.
    The noise level is drawn per episode in [0, task noise], so the collection mixes clean grasps and noisy motion;
    the command is gain times the one that would cover the remaining distance in one step (with gain 2, about half of
    the grasp episodes lift the cube on Panda and Sawyer; with the earlier fixed noise, almost none did).
    A waypoint is (position, gripper, tolerance, min_hold): the next one starts once the gripper is within the
    tolerance and min_hold steps have passed (the gripper needs a few steps to close), or after max_hold steps."""
    low, high = np.array(task["low"]), np.array(task["high"])
    target = obs[task["object"]]
    sigma = rng.uniform(0, task["noise"])
    if rng.random() < 2 / 3:  # reach, grasp, lift
        off = np.r_[rng.normal(0, 0.01, 2), 0]
        wps = [(target + off + [0, 0, 0.08], -1, 0.02, 0), (target + off, -1, 0.01, 0), (target + off, 1, 1.0, 3),
               (target + [0, 0, 0.12], 1, 0.02, 0)]
    else:  # random targets in the workspace
        wps = [(rng.uniform(low, high), rng.choice([-1, 1]), 0.02, 0) for _ in range(5)]
    w, hold, noise = 0, 0, np.zeros(3)
    while True:
        goal, grip, tolerance, min_hold = wps[w]
        delta = np.clip(goal, low, high) - obs["robot0_eef_pos"]
        noise = 0.7 * noise + rng.normal(0, sigma, 3)
        a = np.zeros(7)
        a[:3] = np.clip(delta / (frameskip * 0.05) * task["gain"] * rng.uniform(0.5, 1.0) + noise, -1, 1)
        a[6] = grip
        obs = yield a
        hold += 1
        if (np.linalg.norm(delta) < tolerance and hold >= min_hold) or hold > task["max_hold"]:
            w, hold = min(w + 1, len(wps) - 1), 0


def random_action(rng, obs, task):
    """Uniform random translation and gripper; pushes outside the workspace are cancelled"""
    low, high = np.array(task["low"]), np.array(task["high"])
    a = np.zeros(7)
    a[:3] = rng.uniform(-1, 1, 3)
    a[6] = rng.choice([-1, 1])
    eef = obs["robot0_eef_pos"]
    a[:3][((eef <= low) & (a[:3] < 0)) | ((eef >= high) & (a[:3] > 0))] = 0  # keep the gripper in view
    return a


def random(rng, obs, task, frameskip):
    """random_action in the generator protocol of the collection"""
    while True:
        obs = yield random_action(rng, obs, task)


POLICIES = {"scripted": scripted, "random": random}


def write_video(frames, path, fps, crf):
    """RGB frames -> H.264 mp4, one video frame per given frame"""
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), str(path)]
    subprocess.run(cmd, input=np.stack(frames).tobytes(), check=True)


ENVS = {}  # one environment per robot, task and cameras in each worker process


def episode(args):
    """Plays one episode and writes its folder: steps + 1 frames per camera, steps actions"""
    cfg, robot, name, policy, seed = args
    from h2r.envs.robosuite_env import RobosuiteEnv  # in the worker: each process has its own renderer

    task = cfg["tasks"][name]
    key = (robot, name, *cfg["cameras"], cfg["env"]["camera_heights"], cfg["env"]["camera_widths"])
    if key not in ENVS:
        ENVS[key] = RobosuiteEnv({"env_name": name, "robots": robot, "camera_names": cfg["cameras"], **cfg["env"]})
    env = ENVS[key]
    np.random.seed(seed)  # robosuite places the objects with the global generator
    rng = np.random.default_rng(seed)
    obs = env.reset()
    act = POLICIES[policy](rng, obs, task, cfg["frameskip"])
    a = next(act)
    frames = {cam: [] for cam in cfg["cameras"]}
    rows, sim = [], []
    for t in range(cfg["steps"] + 1):
        for cam in cfg["cameras"]:
            frames[cam].append(env.image(obs, cam))
        rows.append(env.robot_state(obs))
        sim.append({"command": None, "sim_state": env.sim_state().tolist(),
                    "object": obs[task["object"]].tolist(), "eef_world": obs["robot0_eef_pos"].tolist()})
        if t == cfg["steps"]:
            break
        obs = env.step(a, cfg["frameskip"])[0]
        sim[-1]["command"] = a.tolist()
        a = act.send(obs)

    out = Path(cfg["raw_dir"]) / robot / name / policy / f"episode_{seed:06d}"
    out.mkdir(parents=True, exist_ok=True)
    fps = cfg["env"]["control_freq"] / cfg["frameskip"]
    for cam, f in frames.items():
        write_video(f, out / f"{cam}.mp4", fps, cfg["video_crf"])
    times = (np.arange(len(rows)) / fps).tolist()
    pq.write_table(pa.table({"timestamp": times, "state": np.stack(rows).tolist()}), out / "state.parquet")
    pq.write_table(pa.Table.from_pylist([{"timestamp": t} | s for t, s in zip(times, sim)]), out / "sim.parquet")
    return out


def collect(cfg):
    raw = Path(cfg["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg["config_path"], raw / "config.yaml")  # how the episodes were made
    jobs = [(cfg, robot, name, policy, seed) for robot in cfg["robots"] for name in cfg["collect"]
            for policy in cfg["policies"] for seed in range(cfg["episodes"])]
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        for i, _ in enumerate(pool.map(episode, jobs, chunksize=4)):
            if (i + 1) % 100 == 0:
                print(f"{i + 1} / {len(jobs)}", flush=True)
    print(f"{len(jobs)} episodes in {raw}, robots {cfg['robots']}, cameras {cfg['cameras']}")
