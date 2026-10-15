"""Policies in robosuite, and their closed-loop evaluation on the test episodes of h2r data collect.

A policy has reset(episode) and policy(obs, state) -> OSC command (7,); state is the robot state
[x, y, z, roll, pitch, yaw, gripper] (RobosuiteEnv.robot_state). To add one: write such a class, add it to POLICIES
and list it in the config.

Each episode: the simulator is reset to the frame t0 of a collected test episode, the goal is its frame t0 + G, and
the policy acts for M steps (each command held frameskip control steps). Metric: distance of the gripper to its
position at the goal frame, at the start and at the end.
"""

import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image

from h2r.data.prepare import extract_frames, split_of, states_to_actions
from h2r.data.sim import random
from h2r.envs.robosuite_env import RobosuiteEnv
from h2r.plan import context_latents, encode, load, plan
from h2r.training.utils import make_logger, prepare_run, setup


class RandomPolicy:
    """Baseline: uniform random translation and gripper commands, cancelled when they leave the workspace
    (the random policy of h2r data collect)"""

    def __init__(self, cfg, sim_cfg, device):
        self.task, self.frameskip = sim_cfg["tasks"][cfg["task"]], sim_cfg["frameskip"]
        self.rng = np.random.default_rng(cfg["seed"])

    def reset(self, episode):
        self.actions = None

    def __call__(self, obs, state):
        if self.actions is None:
            self.actions = random(self.rng, obs, self.task, self.frameskip)
            return next(self.actions)
        return self.actions.send(obs)


class PlannerPolicy:
    """CEM with the controller in the loop: plans H actions towards the goal frame, sends the first one, replans.
    The context holds the last real frames and the measured actions linking them."""

    def __init__(self, cfg, sim_cfg, device):
        self.cfg, self.p, self.device = cfg, cfg["planner"], device
        self.model, self.ctrl = load(cfg["controller"], device)
        self.command_map = fit_command_map(episode_dirs(sim_cfg, cfg["task"], cfg["episodes_from"], "train"))
        self.gen = torch.Generator().manual_seed(cfg["seed"])

    def reset(self, episode):
        t0, goal_t = episode["t0"], episode["goal_t"]
        self.goal = encode(self.model, episode["frames"][goal_t : goal_t + 1], self.device)[0]
        self.imgs, self.states = list(episode["frames"][:t0]), list(episode["states"][:t0])  # before t0

    def __call__(self, obs, state):
        self.imgs.append(RobosuiteEnv.image(obs, self.cfg["camera"]))
        self.states.append(state)
        context = slice(-self.p["context"], None)
        emb = encode(self.model, self.imgs[context], self.device)
        past = torch.tensor(states_to_actions(np.stack(self.states[context])), dtype=torch.float32, device=self.device)
        latents = context_latents(self.model, self.ctrl, emb, past, self.p["past_latents"])
        action = plan(self.model, self.ctrl, emb, latents, self.goal, self.p, self.gen)[0]
        return to_command(action.cpu().numpy(), state, *self.command_map)


POLICIES = {"random": RandomPolicy, "planner": PlannerPolicy}


def episode_dirs(sim_cfg, task, policy, split):
    """Collected episodes of a task and policy in a split, as h2r data extract assigns it (split_by: folder)"""
    dirs = sorted(d for d in (Path(sim_cfg["raw_dir"]) / task / policy).glob("episode_*") if d.is_dir())
    return [d for d in dirs if split_of(f"{task}/{policy}/{d.name}", sim_cfg["split"]) == split]


def fit_command_map(folders):
    """Least squares per axis: OSC command = gain * measured translation + offset"""
    deltas, commands = [], []
    for f in folders:
        deltas.append(states_to_actions(np.asarray(pq.read_table(f / "state.parquet")["state"].to_pylist()))[:, :3])
        commands.append([s["command"][:3] for s in pq.read_table(f / "sim.parquet").to_pylist()[:-1]])
    deltas, commands = np.concatenate(deltas), np.concatenate(commands)
    gain, offset = np.array([np.polyfit(deltas[:, i], commands[:, i], 1) for i in range(3)]).T
    return gain, offset


def to_command(action, state, gain, offset):
    """Measured-delta action -> OSC command: translation by the fitted map, no rotation, gripper closes past one half"""
    command = np.zeros(7)
    command[:3] = np.clip(gain * action[:3] + offset, -1, 1)
    command[6] = 1.0 if state[6] + action[6] > 0.5 else -1.0
    return command


def load_episode(folder, cfg, sim_cfg):
    """Frames of the policy's camera, robot states and simulator states of a collected episode, start and goal"""
    video = folder / f"{cfg['camera']}.mp4"
    frames = [np.asarray(Image.open(io.BytesIO(j))) for j in extract_frames(video, sim_cfg["fps"], sim_cfg["image_size"])]
    return {"name": folder.name, "frames": frames, "t0": cfg["start"], "goal_t": cfg["start"] + cfg["goal_offset"],
            "states": np.asarray(pq.read_table(folder / "state.parquet")["state"].to_pylist()),
            "sim": pq.read_table(folder / "sim.parquet").to_pylist()}


def run_episode(env, policy, episode, steps, frameskip):
    """Closed loop from the frame t0; returns the start and final distances of the gripper to the goal position"""
    obs = env.restore(episode["sim"][episode["t0"]]["sim_state"])
    policy.reset(episode)
    goal = episode["states"][episode["goal_t"], :3]
    start = float(np.linalg.norm(env.robot_state(obs)[:3] - goal))
    for _ in range(steps):
        obs = env.step(policy(obs, env.robot_state(obs)), frameskip)[0]
    return start, float(np.linalg.norm(env.robot_state(obs)[:3] - goal))


def evaluate(cfg):
    cfg, run_dir, _ = prepare_run(cfg)
    log = make_logger(run_dir)
    device, _ = setup(cfg)
    sim_cfg = yaml.safe_load(open(cfg["sim_config"]))
    policies = {name: POLICIES[name](cfg, sim_cfg, device) for name in cfg["policies"]}
    tests = episode_dirs(sim_cfg, cfg["task"], cfg["episodes_from"], cfg["split"])[: cfg["episodes"]]
    env = RobosuiteEnv({"env_name": cfg["task"], "camera_names": [cfg["camera"]], **sim_cfg["env"]})
    env.reset()  # once: restore() puts each episode's state back
    results = []
    for folder in tests:
        episode = load_episode(folder, cfg, sim_cfg)
        row = {"episode": folder.name}
        for name, policy in policies.items():
            row["start"], row[name] = run_episode(env, policy, episode, cfg["steps"], sim_cfg["frameskip"])
        results.append(row)
        log(f"{folder.name} | start {row['start'] * 100:.1f} cm | "
            + " | ".join(f"{n} {row[n] * 100:.1f} cm" for n in policies))
    summary = {k: float(np.mean([r[k] for r in results])) for k in ["start", *policies]}
    summary |= {f"{n}_closer": int(sum(r[n] < r["start"] for r in results)) for n in policies}
    json.dump({"summary": summary, "episodes": results}, open(run_dir / "results.json", "w"), indent=2)
    log(f"{len(results)} episodes | start {summary['start'] * 100:.1f} cm | " + " | ".join(
        f"{n} {summary[n] * 100:.1f} cm, closer {summary[n + '_closer']}/{len(results)}" for n in policies))
