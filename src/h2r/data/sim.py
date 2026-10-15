"""Collect robosuite episodes into raw/, in the same layout as robot videos, for h2r data extract (actions: true).

raw_dir/<task>/<policy>/episode_<seed>/
  <camera>.mp4    one frame per action (control_freq / frameskip fps), one file per camera of the config
  state.parquet   timestamp + state [x, y, z, roll, pitch, yaw, gripper]: end effector in the robot base frame,
                  Euler angles (xyz) in radians, gripper closedness in [0, 1], as in DROID
  sim.parquet     what only simulation has: commanded OSC actions, MuJoCo states (to reset the simulator on any
                  frame), object and end-effector world positions

Two policies: noisy scripted waypoints (reach, grasp, lift, or random targets) and random actions.
Each action is held for frameskip control steps, so one action per model step.
"""

import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


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


def robot_state(env, obs):
    """[x, y, z, roll, pitch, yaw, gripper] in the robot base frame, as DROID's cartesian and gripper positions"""
    robot = env.robots[0]
    base = env.sim.model.body_name2id(robot.robot_model.root_body)
    base_pos, base_rot = env.sim.data.body_xpos[base], Rotation.from_matrix(env.sim.data.body_xmat[base].reshape(3, 3))
    pos = base_rot.inv().apply(obs["robot0_eef_pos"] - base_pos)
    rot = (base_rot.inv() * Rotation.from_quat(obs["robot0_eef_quat"])).as_euler("xyz")  # robosuite quat: x, y, z, w
    joints = list(robot.gripper.values())[0].joints
    ranges = [env.sim.model.jnt_range[env.sim.model.joint_name2id(j)] for j in joints]
    q = obs["robot0_gripper_qpos"]
    closed = 1 - (q[0] - q[1]) / (ranges[0][1] - ranges[1][0])  # 0: fingers fully apart, 1: touching
    return np.r_[pos, rot, np.clip(closed, 0, 1)]


def write_video(frames, path, fps, crf):
    """RGB frames -> H.264 mp4, one video frame per given frame"""
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), str(path)]
    subprocess.run(cmd, input=np.stack(frames).tobytes(), check=True)


ENVS = {}  # one environment per task in each worker process


def episode(args):
    """Plays one episode and writes its folder: steps + 1 frames per camera, steps actions"""
    cfg, name, policy, seed = args
    import robosuite as suite  # in the worker: each process has its own renderer

    task = cfg["tasks"][name]
    if name not in ENVS:
        ENVS[name] = suite.make(name, camera_names=cfg["cameras"], **cfg["env"])
    env = ENVS[name]
    np.random.seed(seed)  # robosuite places the objects with the global generator
    rng = np.random.default_rng(seed)
    obs = env.reset()
    act = POLICIES[policy](rng, obs, task, cfg["frameskip"])
    a = next(act)
    frames = {cam: [] for cam in cfg["cameras"]}
    rows, sim = [], []
    for t in range(cfg["steps"] + 1):
        for cam in cfg["cameras"]:
            frames[cam].append(obs[f"{cam}_image"][::-1])  # robosuite renders upside down
        rows.append(robot_state(env, obs))
        sim.append({"command": None, "sim_state": env.sim.get_state().flatten().tolist(),
                    "object": obs[task["object"]].tolist(), "eef_world": obs["robot0_eef_pos"].tolist()})
        if t == cfg["steps"]:
            break
        for _ in range(cfg["frameskip"]):
            obs, *_ = env.step(a)
        sim[-1]["command"] = a.tolist()
        a = act.send(obs)

    out = Path(cfg["raw_dir"]) / name / policy / f"episode_{seed:06d}"
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
    jobs = [(cfg, name, policy, seed) for name in cfg["tasks"] for policy in cfg["policies"]
            for seed in range(cfg["episodes"])]
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        for i, _ in enumerate(pool.map(episode, jobs, chunksize=4)):
            if (i + 1) % 100 == 0:
                print(f"{i + 1} / {len(jobs)}", flush=True)
    print(f"{len(jobs)} episodes in {raw}, cameras {cfg['cameras']}")
