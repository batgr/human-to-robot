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
    root = Path(sim_cfg["raw_dir"]) / robot / task / policy
    episodes = sorted(path for path in root.glob("episode_*") if path.is_dir())
    return [
        path for path in episodes
        if split_of(f"{robot}/{task}/{policy}/{path.name}", sim_cfg["split"]) == split
    ]


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
    # Use the first closing motion as pre-grasp and its first subsequent
    # near-zero gripper change as grasp. If no closing starts, use the lift frame.
    gripper_changes = np.diff(states[:, 6])
    closing = np.flatnonzero(gripper_changes > 0.05)
    pre_grasp = int(closing[0]) if len(closing) else top

    grasp = top
    if pre_grasp + 1 < top:
        settled = np.diff(states[pre_grasp + 1 :, 6]) < 0.01
        grasp = pre_grasp + 1 + int(settled.argmax())

    first_goal = t0 + 1
    return [max(pre_grasp, first_goal), max(min(grasp, top), first_goal), top]


def load_episode(folder, cfg, sim_cfg):
    """Frames of the camera, robot states [x, y, z, roll, pitch, yaw, gripper], simulator states, t0 and goal frames"""
    video = folder / f"{cfg['camera']}.mp4"
    frames = []
    for jpeg in extract_frames(video, sim_cfg["fps"], sim_cfg["image_size"]):
        with Image.open(io.BytesIO(jpeg)) as image:
            frames.append(np.asarray(image))
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
    """render: true opens a window on the camera the world model sees (robosuite's OpenCV viewer: its native viewer
    needs mjpython on macOS, where the offscreen renderer the planner uses crashes)"""
    render_options = {}
    if cfg.get("render"):
        render_options = {"has_renderer": True, "renderer": "mujoco", "render_camera": cfg["camera"]}

    env_cfg = {
        "env_name": cfg["task"],
        "robots": cfg["robot"],
        "camera_names": [cfg["camera"]],
        **sim_cfg["env"],
        **render_options,
    }
    env = RobosuiteEnv(env_cfg)
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

    gripper_delta = action[6]
    if abs(gripper_delta) > 0.05:
        close = gripper_delta > 0
    else:
        close = state[6] > 0.25
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
