"""Planning with the cross-entropy method (CEM), and closed-loop episodes in robosuite (h2r plan run)

Each step, the policy gives an OSC delta command, held frameskip control steps as in the collection:
- scripted, random: the collection's policies (same seed: the recorded episode again);
- recorded, zero, cem: an action of the data (end-effector move in the base frame, relative rotation, gripper closure
  change), turned into a command by to_command. The CEM searches the next actions through the world model and the
  controller, from the frames of the episode passed through the data chain, toward a goal frame of the recording.
"""

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from huggingface_hub import hf_hub_download

from h2r.data.dataset import decode, shards
from h2r.data.prepare import data_frame, states_to_actions, write_video


def goal_cost(model, controller, context, latents, goal):
    """Cost of action plans: distance between the last imagined state and the goal
    context: (B, C, D) real states, latents: (B, C - 1, z_dim) the latent actions linking them, goal: (B, D)
    returns cost(plans (B, S, H, act_dim)) -> (B, S)"""

    def cost(plans):
        final = model.rollout(context, latents, plans, controller)[:, :, -1]
        return ((final - goal[:, None]) ** 2).mean(-1)

    return cost


def cem(cost, mean, std, samples=300, elites=30, iterations=10):
    """Sample plans from a Gaussian, keep the cheapest, refit the Gaussian on them, repeat
    cost: plans (B, S, H, A) -> (B, S); mean, std: (B, H, A) the starting Gaussian
    returns the final mean (B, H, A), and at each iteration the mean cost of the elites (iterations, B) and the mean
    (iterations, B, H, A)"""
    elite_costs, means = [], []
    for _ in range(iterations):
        plans = mean[:, None] + std[:, None] * torch.randn(mean.shape[0], samples, *mean.shape[1:], device=mean.device)
        costs = cost(plans)
        best = costs.topk(elites, dim=1, largest=False).indices  # (B, elites)
        elite = plans[torch.arange(len(plans))[:, None], best]  # (B, elites, H, A)
        mean, std = elite.mean(1), elite.std(1)
        elite_costs.append(costs.gather(1, best).mean(1))
        means.append(mean)
    return mean, torch.stack(elite_costs), torch.stack(means)


def to_command(action, previous, K, threshold):
    """Action of the data (7,) -> OSC delta command (7,): move / K clipped to [-1, 1] (K: achieved move per unit of
    command, fitted on Sim train, experimentation.ipynb 9 Nov.); rotation 0, as the collection's script; the gripper
    closes (1) or opens (-1) when the closure must change by more than threshold, else keeps the previous command"""
    command = np.zeros(7)
    command[:3] = np.clip(np.asarray(action[:3]) / np.asarray(K), -1, 1)
    command[6] = 1.0 if action[6] > threshold else -1.0 if action[6] < -threshold else previous[6]
    return command


def scripted(rng, obs, task, frameskip):
    """The collection's scripted policy (1ac8d37^:src/h2r/data/sim.py), a generator of commands toward noisy waypoints
    that receives the new observation after each command. Noise level drawn per episode in [0, task noise]; the command
    is gain times the one that would cover the remaining distance in one step. A waypoint is (position, gripper,
    tolerance, min_hold): the next one starts once the gripper is within the tolerance and min_hold steps have passed,
    or after max_hold steps."""
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


def random(rng, obs, task, frameskip):
    """The collection's random policy: uniform translation and gripper; pushes outside the workspace are cancelled"""
    low, high = np.array(task["low"]), np.array(task["high"])
    while True:
        a = np.zeros(7)
        a[:3] = rng.uniform(-1, 1, 3)
        a[6] = rng.choice([-1, 1])
        eef = obs["robot0_eef_pos"]
        a[:3][((eef <= low) & (a[:3] < 0)) | ((eef >= high) & (a[:3] > 0))] = 0  # keep the gripper in view
        obs = yield a


COLLECTION_POLICIES = {"scripted": scripted, "random": random}


def stages(env, obs, measures):
    """Which stages are met now: reach (gripper centre near the cube centre), grasp (robosuite's test), lift"""
    return {"reach": bool(np.linalg.norm(obs["robot0_eef_pos"] - obs["cube_pos"]) < measures["reach"]),
            "grasp": bool(env.grasping()),
            "lift": bool(obs["cube_pos"][2] > env.table_height() + measures["lift"])}


def stage_times(env, demo, cfg):
    """First step of the recording where each stage is met, from an exact replay of its OSC commands (the simulation
    is deterministic, experimentation.ipynb 9 Nov.); a stage never met is absent"""
    obs = env.reset(demo["seed"])
    first = {}
    for t in range(len(demo["commands"]) + 1):
        if t > 0:
            for _ in range(cfg["frameskip"]):
                obs = env.step(np.asarray(demo["commands"][t - 1]))
        for k, met in stages(env, obs, cfg["measures"]).items():
            if met:
                first.setdefault(k, t)
    return first


def goal_schedule(kind, goals, steps, k=None, times=None):
    """Goal state at each step (steps, D), from the states of the recorded frames goals (T, D):
    final: the last frame; dense: frame t + k (the last one past the end); stages: the frame where the recording first
    reaches, then the one where it first grasps, then the last frame, each one until the recording reaches that stage"""
    last = len(goals) - 1
    if kind == "final":
        idx = [last] * steps
    elif kind == "dense":
        idx = [min(t + k, last) for t in range(steps)]
    elif kind == "stages":
        idx = [times["reach"] if t < times["reach"] else times["grasp"] if t < times["grasp"] else last
               for t in range(steps)]
    else:
        raise ValueError(f"policy.goal: final, dense or stages, got {kind}")
    return goals[idx]


def demonstrations(cfg, env):
    """The recorded episodes the closed loop starts from: [{"video", "seed", "jpegs" (T,), "actions" (T - 1, 7),
    "commands" (T - 1, 7) OSC commands sent, "states" (T, 7) robot states, "cube" (T, 3)}]"""
    e = cfg["episodes"]
    out = []
    for shard in shards(e["root"], e["split"]):
        for r in pq.read_table(shard, columns=["video", "frames", "actions"]).to_pylist():
            if f"/{e['recorded_by']}/" not in r["video"]:
                continue
            folder = "raw/" + r["video"].rsplit("/", 1)[0]  # raw/<robot>/<task>/<recorded_by>/episode_<seed>
            raw = {name: pq.read_table(hf_hub_download(e["hf_repo"], f"{folder}/{name}", repo_type="dataset",
                                                       local_dir=e["root"])).to_pylist()
                   for name in ("sim.parquet", "state.parquet")}
            cube = np.array([s["object"] for s in raw["sim.parquet"]])
            if e["lifted_only"] and not (cube[:, 2] > env.table_height() + cfg["measures"]["lift"]).any():
                continue
            out.append({"video": r["video"], "seed": int(folder.rsplit("_", 1)[1]), "jpegs": r["frames"],
                        "actions": np.array(r["actions"]), "commands": [s["command"] for s in raw["sim.parquet"][:-1]],
                        "states": np.array([s["state"] for s in raw["state.parquet"]]), "cube": cube})
            if len(out) == e["count"]:
                return out
    return out


def encode(model, frames, device):
    """(N, 3, H, W) uint8 frames -> (N, D) states, each frame a clip of one"""
    with torch.no_grad():
        return model.encode(frames[:, None].to(device))[:, 0].float()


def plan_step(model, controller, states, actions, goal, p):
    """CEM over the next p["horizon"] actions of the data. states: the encoded states so far, actions: the achieved
    actions between them. Context: the last p["context"] states, linked by the controller's z of the achieved actions;
    before the start, the first state repeated and a zero action. -> plan (H, 7), final mean cost of the elites"""
    c, h = p["context"], p["horizon"]
    past_s = [states[0]] * (c - 1) + states
    past_a = [np.zeros(7)] * (c - 1) + actions
    with torch.no_grad():
        s = torch.stack(past_s[-c:])[None]  # (1, C, D)
        a = torch.tensor(np.array(past_a[len(past_a) - (c - 1):]).reshape(-1, 7), dtype=torch.float32)
        z = controller(a[None].to(s.device), s[:, :-1])  # (1, C - 1, z_dim)
        mean = controller.act_mean.expand(1, h, -1).clone()
        std = controller.act_std.expand(1, h, -1).clone()
        plan, costs, _ = cem(goal_cost(model, controller, s, z, goal[None]), mean, std,
                             p["samples"], p["elites"], p["iterations"])
    return plan[0].cpu().numpy(), costs[-1, 0].item()


def episode(env, cfg, demo, model=None, controller=None, device=None):
    """One closed-loop episode from the recording's seed -> per-step log and stage measures"""
    p, ex = cfg["policy"], cfg["executor"]
    torch.manual_seed(demo["seed"])
    rng = np.random.default_rng(demo["seed"])  # as the collection: the scripted policy replays the recording
    obs = env.reset(demo["seed"])
    command = np.r_[np.zeros(6), -1.0]  # gripper open
    fps = env.cfg["control_freq"] / cfg["frameskip"]
    see = lambda: decode([data_frame(env.frame(), fps, ex["video_crf"], ex["image_size"])])

    if p["type"] in COLLECTION_POLICIES:
        policy = COLLECTION_POLICIES[p["type"]](rng, obs, cfg["task"], cfg["frameskip"])
        command = next(policy)
    if p["type"] == "cem":
        recorded = encode(model, decode(demo["jpegs"]), device)
        times = stage_times(env, demo, cfg) if p["goal"] == "stages" else None
        goals = goal_schedule(p["goal"], recorded, cfg["steps"], p.get("k"), times)
        obs = env.reset(demo["seed"])  # stage_times replayed the recording
        states, achieved = [encode(model, see(), device)[0]], []

    robot = [env.robot_state(obs)]
    log = {"stages": [stages(env, obs, cfg["measures"])], "distance": [float(np.linalg.norm(obs["robot0_eef_pos"] - obs["cube_pos"]))],
           "cube": [obs["cube_pos"].tolist()], "gripper": [float(command[6])], "plan": [], "cost": [], "time": []}
    frames = [env.frame()] if cfg["render"]["video"] else None
    for t in range(cfg["steps"]):
        start = time.time()
        if p["type"] in COLLECTION_POLICIES:
            command = policy.send(obs) if t > 0 else command
        elif p["type"] == "recorded":
            command = to_command(demo["actions"][t], command, ex["K"], ex["gripper_threshold"])
        elif p["type"] == "zero":
            command = to_command(np.zeros(7), command, ex["K"], ex["gripper_threshold"])
        elif p["type"] == "cem":
            plan, cost = plan_step(model, controller, states, achieved, goals[t], p)
            command = to_command(plan[0], command, ex["K"], ex["gripper_threshold"])
            log["plan"].append(plan[0].tolist())
            log["cost"].append(cost)
        else:
            raise ValueError(f"policy.type: random, scripted, recorded, zero or cem, got {p['type']}")
        log["time"].append(time.time() - start)
        for _ in range(cfg["frameskip"]):
            obs = env.step(command)
        robot.append(env.robot_state(obs))
        if p["type"] == "cem":
            states.append(encode(model, see(), device)[0])
            achieved.append(states_to_actions(np.stack(robot[-2:]))[0])  # the achieved action, as in the data
        log["stages"].append(stages(env, obs, cfg["measures"]))
        log["distance"].append(float(np.linalg.norm(obs["robot0_eef_pos"] - obs["cube_pos"])))
        log["cube"].append(obs["cube_pos"].tolist())
        log["gripper"].append(float(command[6]))
        if frames is not None:
            frames.append(env.frame())

    first = {k: next((t for t, s in enumerate(log["stages"]) if s[k]), None) for k in ("reach", "grasp", "lift")}
    result = {"video": demo["video"], "seed": demo["seed"], **{k: v is not None for k, v in first.items()},
              "first": first, "grasped_steps": sum(s["grasp"] for s in log["stages"]),
              "closings": int((np.diff(log["gripper"]) > 0).sum()),
              "final_gripper_error": float(np.linalg.norm(robot[-1][:3] - demo["states"][-1][:3])),
              "final_cube_error": float(np.linalg.norm(np.array(log["cube"][-1]) - demo["cube"][-1])),
              "time_per_step": float(np.mean(log["time"])), "robot": np.array(robot).tolist(), **log}
    return result, frames


def wilson(k, n, z=1.96):
    """95 % interval of a success rate k / n"""
    if n == 0:
        return [0.0, 1.0]
    p, d = k / n, 1 + z ** 2 / n
    centre, half = (p + z ** 2 / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def summarize(results):
    """Per stage: count, rate and its interval, median step it is first met; medians of the other measures"""
    n = len(results)
    out = {"episodes": n}
    for k in ("reach", "grasp", "lift"):
        hits = [r["first"][k] for r in results if r[k]]
        out[k] = {"count": len(hits), "rate": len(hits) / n, "interval": wilson(len(hits), n),
                  "median_first_step": float(np.median(hits)) if hits else None}
    for k in ("grasped_steps", "closings", "final_gripper_error", "final_cube_error", "time_per_step"):
        out[k] = float(np.median([r[k] for r in results]))
    return out


def run_name(cfg):
    p = cfg["policy"]
    if p["type"] != "cem":
        return p["type"]
    goal = f"dense{p['k']}" if p["goal"] == "dense" else p["goal"]
    return f"cem-{goal}-h{p['horizon']}-c{p['context']}"


def run(cfg):
    """h2r plan run: one closed-loop episode per recorded episode; run folder with the config, episodes.jsonl (one line
    per episode, with its per-step log), summary.json and the videos"""
    from h2r.envs.robosuite_env import RobosuiteEnv  # robosuite only for this command

    out = Path(cfg["output_dir"]) / (cfg["name"] or run_name(cfg))
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"), sort_keys=False)
    env = RobosuiteEnv(cfg["env"])
    model = controller = device = None
    if cfg["policy"]["type"] == "cem":
        from h2r.models.controller import load_controller
        from h2r.training.utils import pick_device

        device = pick_device()
        model, controller = load_controller(cfg["policy"]["controller"], device)
        model.history_size = cfg["policy"]["context"]
    demos = demonstrations(cfg, env)
    results = []
    with open(out / "episodes.jsonl", "w") as f:
        for i, demo in enumerate(demos):
            result, frames = episode(env, cfg, demo, model, controller, device)
            if frames is not None:
                (out / "videos").mkdir(exist_ok=True)
                write_video(frames, out / "videos" / f"episode_{demo['seed']:06d}.mp4",
                            env.cfg["control_freq"] / cfg["frameskip"], cfg["render"]["video_crf"])
            results.append(result)
            f.write(json.dumps(result) + "\n")
            print(f"{i + 1} / {len(demos)} episode {demo['seed']}: reach {result['reach']}, grasp {result['grasp']}, "
                  f"lift {result['lift']}, {result['time_per_step']:.2f} s per step", flush=True)
    summary = summarize(results)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(f"{out}: {summary['episodes']} episodes")
    for k in ("reach", "grasp", "lift"):
        s = summary[k]
        print(f"  {k:6s} {s['count']:3d} [{100 * s['interval'][0]:.0f}-{100 * s['interval'][1]:.0f} %], "
              f"first at step {s['median_first_step']}")
    print(f"  final error: gripper {100 * summary['final_gripper_error']:.1f} cm, cube {100 * summary['final_cube_error']:.1f} cm; "
          f"{summary['closings']:.0f} closings, {summary['time_per_step']:.2f} s per step")
    return summary
