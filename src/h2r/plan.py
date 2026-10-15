"""Plan with a trained controller: CEM over real actions, scored by JEPA.get_cost, and two evaluations.

h2r plan offline: on clips of test episodes, plan from t towards the frame t + H and compare the summed planned
translation with the real one (Δxyz, Garrido et al. 2026, appendix A).
h2r plan robosuite: closed loop on test episodes of h2r data collect; reset the simulator to t0, take the frame
t0 + G as goal, plan, execute the first action, observe and replan. Planned actions are measured end-effector deltas;
a linear map fitted on training episodes turns them into OSC commands. Metric: distance of the gripper to its
position at the goal frame, for the planner and for random actions.

The context holds the last real frames and the latent actions linking them: the controller on the executed (measured)
actions, as for the future, or the IDM on the real pairs (past_latents: idm).
"""

import io
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image

from h2r.data.prepare import extract_frames, split_of, states_to_actions
from h2r.data.sim import robot_state
from h2r.training.controller import build_controller, load_world_model, read_episodes
from h2r.training.utils import make_logger, prepare_run, setup


def load(controller_run, device):
    """Controller of a run and the frozen world model it was trained for"""
    run = Path(controller_run)
    cfg = yaml.safe_load(open(run / "config.yaml"))
    model, wm_cfg = load_world_model(cfg["world_model"], device)
    ctrl = build_controller(cfg["model"], wm_cfg).to(device)
    ctrl.load_state_dict(torch.load(run / "best.pt", map_location=device)["model"])
    return model, ctrl.eval().requires_grad_(False)


@torch.no_grad()
def encode(model, frames, device):
    """(T, H, W, 3) uint8 -> (T, N, D)"""
    return model.encode({"pixels": torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].to(device)})["emb"][0]


@torch.no_grad()
def context_latents(model, ctrl, emb, actions, source):
    """Latent actions between the context states: emb (C, N, D), actions (C - 1, act_dim) -> (C - 1, d_a)"""
    if len(emb) < 2:
        return emb.new_zeros(0, ctrl.head.out_features)
    if source == "idm":
        return model.idm(emb[None])[0][0]
    return ctrl(actions, emb[:-1])


@torch.no_grad()
def plan(model, ctrl, emb, latents, goal, cfg, generator=None):
    """CEM in the controller's standardized action space (mean 0, std 1 at the start), searched dimensions only.
    Returns (H, act_dim) actions in physical units."""
    dim = ctrl.act_mean.numel()
    searched = torch.zeros(dim, dtype=torch.bool, device=goal.device)
    searched[cfg["searched"]] = True
    mu, sigma = torch.zeros(cfg["horizon"], dim, device=goal.device), torch.ones(cfg["horizon"], dim, device=goal.device)

    def physical(u):
        a = (u * ctrl.act_std + ctrl.act_mean) * searched  # rotations etc. stay at 0
        if cfg.get("max_norm"):
            a[..., :3] *= (cfg["max_norm"] / a[..., :3].norm(dim=-1, keepdim=True)).clamp(max=1)
        return a

    for _ in range(cfg["iterations"]):
        u = mu + sigma * torch.randn(cfg["samples"], cfg["horizon"], dim, generator=generator).to(goal.device)
        cost = model.get_cost(emb[None], latents[None], goal[None], physical(u)[None], ctrl, cfg["window"])[0]
        elites = u[cost.topk(cfg["elites"], largest=False).indices]
        mu, sigma = elites.mean(0), elites.std(0) + 1e-3
    return physical(u[cost.argmin()] if cfg["pick"] == "best" else mu)


# ---------------------------------------------------------------- offline (Δxyz)


def delta_xyz(planned, real):
    """L1 distance between the summed planned and real translations"""
    return float(np.abs(np.asarray(planned)[:, :3].sum(0) - np.asarray(real)[:, :3].sum(0)).sum())


def offline(cfg):
    cfg, run_dir, _ = prepare_run(cfg)
    log = make_logger(run_dir)
    device, _ = setup(cfg)
    model, ctrl = load(cfg["controller"], device)
    p, gen = cfg["planner"], torch.Generator().manual_seed(cfg["seed"])
    episodes = read_episodes(cfg["data"]["root"], cfg["data"]["split"])
    starts = [(e, t) for e, ep in enumerate(episodes) for t in range(p["context"] - 1, len(ep["frames"]) - p["horizon"])]
    results = []
    for e, t in random.Random(cfg["seed"]).sample(starts, min(cfg["clips"], len(starts))):
        ep = episodes[e]
        idx = list(range(t - p["context"] + 1, t + 1)) + [t + p["horizon"]]
        emb = encode(model, [np.asarray(Image.open(io.BytesIO(ep["frames"][i]))) for i in idx], device)
        past = torch.tensor(ep["actions"][idx[0] : t], dtype=torch.float32, device=device)
        latents = context_latents(model, ctrl, emb[:-1], past, p["past_latents"])
        actions = plan(model, ctrl, emb[:-1], latents, emb[-1], p, gen).cpu()
        real = ep["actions"][t : t + p["horizon"]]
        results.append({"video": ep["video"], "t": t, "delta_xyz": delta_xyz(actions, real),
                        "no_motion": delta_xyz(np.zeros_like(real), real)})
    summary = {k: float(np.mean([r[k] for r in results])) for k in ("delta_xyz", "no_motion")}
    json.dump({"summary": summary, "clips": results}, open(run_dir / "results.json", "w"), indent=2)
    log(f"{len(results)} clips | Δxyz {summary['delta_xyz']:.4f} (no motion {summary['no_motion']:.4f})")


# ---------------------------------------------------------------- robosuite (closed loop)


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
    """Planned delta -> OSC command: translation by the fitted map, no rotation, gripper closes past one half"""
    command = np.zeros(7)
    command[:3] = np.clip(gain * action[:3] + offset, -1, 1)
    command[6] = 1.0 if state[6] + action[6] > 0.5 else -1.0
    return command


def run_episode(env, model, ctrl, folder, cfg, sim_cfg, command_map, device, gen, random_actions=False):
    """One closed-loop episode; returns the start and final distances of the gripper to the goal position"""
    p, cam = cfg["planner"], cfg["camera"]
    frames = [np.asarray(Image.open(io.BytesIO(j)))
              for j in extract_frames(folder / f"{cam}.mp4", sim_cfg["fps"], sim_cfg["image_size"])]
    states = np.asarray(pq.read_table(folder / "state.parquet")["state"].to_pylist())
    sim = pq.read_table(folder / "sim.parquet").to_pylist()
    t0 = p["context"] - 1
    goal_pos, goal = states[t0 + cfg["goal_offset"], :3], encode(model, frames[t0 + cfg["goal_offset"] :][:1], device)[0]
    env.reset()
    env.sim.set_state_from_flattened(np.asarray(sim[t0]["sim_state"]))
    env.sim.forward()
    obs = env._get_observations(force_update=True)
    imgs, robot = frames[: t0 + 1], list(states[: t0 + 1])  # context of the episode, then observed frames and states
    start = float(np.linalg.norm(robot_state(env, obs)[:3] - goal_pos))
    searched = torch.zeros_like(ctrl.act_mean)
    searched[p["searched"]] = 1
    for _ in range(cfg["steps"]):
        if random_actions:  # same distribution as the CEM's first samples
            action = (torch.randn(len(searched), generator=gen).to(device) * ctrl.act_std + ctrl.act_mean) * searched
        else:
            emb = encode(model, imgs[-p["context"] :], device)
            past = torch.tensor(states_to_actions(np.stack(robot[-p["context"] :])), dtype=torch.float32, device=device)
            latents = context_latents(model, ctrl, emb, past, p["past_latents"])
            action = plan(model, ctrl, emb, latents, goal, p, gen)[0]
        command = to_command(action.cpu().numpy(), robot[-1], *command_map)
        for _ in range(sim_cfg["frameskip"]):
            obs, *_ = env.step(command)
        imgs.append(obs[f"{cam}_image"][::-1].copy())  # robosuite renders upside down
        robot.append(robot_state(env, obs))  # measured: what the robot did, as in the controller's training
    return start, float(np.linalg.norm(robot[-1][:3] - goal_pos))


def robosuite(cfg):
    cfg, run_dir, _ = prepare_run(cfg)
    log = make_logger(run_dir)
    device, _ = setup(cfg)
    sim_cfg = yaml.safe_load(open(cfg["sim_config"]))
    model, ctrl = load(cfg["controller"], device)
    command_map = fit_command_map(episode_dirs(sim_cfg, cfg["task"], cfg["policy"], "train"))
    tests = episode_dirs(sim_cfg, cfg["task"], cfg["policy"], cfg["split"])[: cfg["episodes"]]

    import robosuite as suite

    env = suite.make(cfg["task"], camera_names=[cfg["camera"]], **sim_cfg["env"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    results = []
    for folder in tests:
        start, final = run_episode(env, model, ctrl, folder, cfg, sim_cfg, command_map, device, gen)
        _, rand = run_episode(env, model, ctrl, folder, cfg, sim_cfg, command_map, device, gen, random_actions=True)
        results.append({"episode": folder.name, "start": start, "plan": final, "random": rand})
        log(f"{folder.name} | start {start * 100:.1f} cm | plan {final * 100:.1f} cm | random {rand * 100:.1f} cm")
    summary = {k: float(np.mean([r[k] for r in results])) for k in ("start", "plan", "random")}
    summary["plan_closer"] = int(sum(r["plan"] < r["start"] for r in results))
    json.dump({"summary": summary, "command_map": [v.tolist() for v in command_map], "episodes": results},
              open(run_dir / "results.json", "w"), indent=2)
    log(f"{len(results)} episodes | start {summary['start'] * 100:.1f} cm | plan {summary['plan'] * 100:.1f} cm | "
        f"random {summary['random'] * 100:.1f} cm | plan closer {summary['plan_closer']}/{len(results)}")
