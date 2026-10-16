"""What the evaluations share: the collected test episodes and their goal frames, the simulator restored to them,
and the command map between the controller's actions (measured end-effector deltas) and robosuite's OSC commands."""

import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from h2r.data.prepare import extract_frames, split_of, states_to_actions
from h2r.envs.robosuite_env import RobosuiteEnv
from h2r.plan import context_latents, encode


def episode_dirs(sim_cfg, robot, task, policy, split):
    """Collected episodes of a robot, task and policy in a split, as h2r data extract assigns it (split_by: folder)"""
    dirs = sorted(d for d in (Path(sim_cfg["raw_dir"]) / robot / task / policy).glob("episode_*") if d.is_dir())
    return [d for d in dirs if split_of(f"{robot}/{task}/{policy}/{d.name}", sim_cfg["split"]) == split]


def goal_frames(states, sim, cfg):
    """Goal frame indices of an episode, from the frame t0 = cfg["start"]:
    - goal: offset   the frame t0 + goal_offset;
    - goal: lift     the frame where the object is highest, if it was lifted 4 cm (robosuite's Lift threshold), else
                     None; with subgoals, the pre-grasp and grasp frames before it (V-JEPA 2-AC)"""
    t0 = cfg["start"]
    if cfg["goal"] == "offset":
        return [t0 + cfg["goal_offset"]]
    height = np.array([s["object"][2] for s in sim])
    top = int(height.argmax())
    if top <= t0 or height[top] - height[:3].min() < 0.04:  # objects spawn slightly above the table and settle
        return None
    if not cfg.get("subgoals"):
        return [top]
    closing = np.diff(states[:, 6]) > 0.05  # the gripper closes between frame t and t + 1
    pre = int(closing.argmax()) if closing.any() else top
    closed = pre + 1 + int(np.argmax(np.diff(states[pre + 1 :, 6]) < 0.01)) if pre + 1 < top else top
    return [max(pre, t0 + 1), max(min(closed, top), t0 + 1), top]


def load_episode(folder, cfg, sim_cfg):
    """Frames of the camera, robot states [x, y, z, roll, pitch, yaw, gripper], simulator states, t0 and goal frames"""
    video = folder / f"{cfg['camera']}.mp4"
    frames = [np.asarray(Image.open(io.BytesIO(j))) for j in extract_frames(video, sim_cfg["fps"], sim_cfg["image_size"])]
    states = np.asarray(pq.read_table(folder / "state.parquet")["state"].to_pylist())
    sim = pq.read_table(folder / "sim.parquet").to_pylist()
    return {"name": folder.name, "frames": frames, "t0": cfg["start"], "goals": goal_frames(states, sim, cfg),
            "states": states, "sim": sim}


def held_out_episodes(cfg, sim_cfg):
    """The first cfg["episodes"] test episodes that can serve the goal: initial states never used in training"""
    out = []
    for folder in episode_dirs(sim_cfg, cfg["robot"], cfg["task"], cfg["episodes_from"], "test"):
        if len(out) == cfg["episodes"]:
            break
        ep = load_episode(folder, cfg, sim_cfg)
        if ep["goals"] is not None:
            out.append(ep)
    if not out:
        raise ValueError(f"no test episode of {cfg['robot']}/{cfg['task']}/{cfg['episodes_from']} can serve "
                         f"goal: {cfg['goal']} (with goal: lift, none where the object was lifted)")
    return out


def make_env(cfg, sim_cfg):
    env = RobosuiteEnv({"env_name": cfg["task"], "robots": cfg["robot"], "camera_names": [cfg["camera"]],
                        **sim_cfg["env"]})
    env.reset()  # once: restore() puts each episode's state back
    return env


def fit_command_map(folders):
    """Least squares per axis: OSC command = gain * measured translation + offset"""
    deltas, commands = [], []
    for f in folders:
        deltas.append(states_to_actions(np.asarray(pq.read_table(f / "state.parquet")["state"].to_pylist()))[:, :3])
        commands.append([s["command"][:3] for s in pq.read_table(f / "sim.parquet").to_pylist()[:-1]])
    deltas, commands = np.concatenate(deltas), np.concatenate(commands)
    gain, offset = np.array([np.polyfit(deltas[:, i], commands[:, i], 1) for i in range(3)]).T
    return gain, offset


def command_map(cfg, sim_cfg):
    """Command map fit on the training episodes of the robot and task"""
    return fit_command_map(episode_dirs(sim_cfg, cfg["robot"], cfg["task"], cfg["episodes_from"], "train"))


def to_command(action, state, gain, offset):
    """Measured-delta action -> OSC command: translation by the fitted map, no rotation. Gripper: closes or opens in
    the direction of the closedness change (about ±0.3 per step), otherwise keeps its state: closed above 0.25 (open is
    about 0, closed on the cube about 0.5, closed empty 1)"""
    command = np.zeros(7)
    command[:3] = np.clip(gain * action[:3] + offset, -1, 1)
    close = action[6] > 0 if abs(action[6]) > 0.05 else state[6] > 0.25
    command[6] = 1.0 if close else -1.0
    return command


def play(env, ep, actions, cmap, frameskip, camera=None):
    """Restores the episode's frame t0 and plays the actions. Returns the last observation and, with a camera, the
    frame after each action"""
    obs = env.restore(ep["sim"][ep["t0"]]["sim_state"])
    frames = []
    for a in actions:
        obs = env.step(to_command(a, env.robot_state(obs), *cmap), frameskip)[0]
        if camera:
            frames.append(env.image(obs, camera))
    return obs, frames


@torch.no_grad()
def planner_context(model, ctrl, env, obs, ep, cfg, device):
    """States and latent actions of the planner's context at the restored frame t0, as the planner policy sees them"""
    t0, C = ep["t0"], cfg["planner"]["context"]
    imgs = list(ep["frames"][t0 - C + 1 : t0]) + [env.image(obs, cfg["camera"])]
    states = np.stack(list(ep["states"][t0 - C + 1 : t0]) + [env.robot_state(obs)])
    emb = encode(model, imgs, device)
    past = torch.tensor(states_to_actions(states), dtype=torch.float32, device=device)
    return emb, context_latents(model, ctrl, emb, past, cfg["planner"]["past_latents"])
