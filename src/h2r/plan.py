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
        plans = mean[:, None] + std[:, None] * torch.randn(
            mean.shape[0], samples, *mean.shape[1:], device=mean.device
        )
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
    closes (1) or opens (-1) when the closure must change by more than threshold, else keeps the previous command
    """
    command = np.zeros(7)
    command[:3] = np.clip(np.asarray(action[:3]) / np.asarray(K), -1, 1)
    command[6] = (
        1.0
        if action[6] > threshold
        else -1.0 if action[6] < -threshold else previous[6]
    )
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
        wps = [
            (target + off + [0, 0, 0.08], -1, 0.02, 0),
            (target + off, -1, 0.01, 0),
            (target + off, 1, 1.0, 3),
            (target + [0, 0, 0.12], 1, 0.02, 0),
        ]
    else:  # random targets in the workspace
        wps = [(rng.uniform(low, high), rng.choice([-1, 1]), 0.02, 0) for _ in range(5)]
    w, hold, noise = 0, 0, np.zeros(3)
    while True:
        goal, grip, tolerance, min_hold = wps[w]
        delta = np.clip(goal, low, high) - obs["robot0_eef_pos"]
        noise = 0.7 * noise + rng.normal(0, sigma, 3)
        a = np.zeros(7)
        a[:3] = np.clip(
            delta / (frameskip * 0.05) * task["gain"] * rng.uniform(0.5, 1.0) + noise,
            -1,
            1,
        )
        a[6] = grip
        obs = yield a
        hold += 1
        if (np.linalg.norm(delta) < tolerance and hold >= min_hold) or hold > task[
            "max_hold"
        ]:
            w, hold = min(w + 1, len(wps) - 1), 0


def random(rng, obs, task, frameskip):
    """The collection's random policy: uniform translation and gripper; pushes outside the workspace are cancelled"""
    low, high = np.array(task["low"]), np.array(task["high"])
    while True:
        a = np.zeros(7)
        a[:3] = rng.uniform(-1, 1, 3)
        a[6] = rng.choice([-1, 1])
        eef = obs["robot0_eef_pos"]
        a[:3][
            ((eef <= low) & (a[:3] < 0)) | ((eef >= high) & (a[:3] > 0))
        ] = 0  # keep the gripper in view
        obs = yield a


COLLECTION_POLICIES = {"scripted": scripted, "random": random}


def stages(env, obs, measures):
    """Which stages are met now: reach (gripper centre near the cube centre), grasp (robosuite's test), lift"""
    return {
        "reach": bool(
            np.linalg.norm(obs["robot0_eef_pos"] - obs["cube_pos"]) < measures["reach"]
        ),
        "grasp": bool(env.grasping()),
        "lift": bool(obs["cube_pos"][2] > env.table_height() + measures["lift"]),
    }


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
    reaches, then the one where it first grasps, then the last frame, each one until the recording reaches that stage
    """
    last = len(goals) - 1
    if kind == "final":
        idx = [last] * steps
    elif kind == "dense":
        idx = [min(t + k, last) for t in range(steps)]
    elif kind == "stages":
        idx = [
            (
                times["reach"]
                if t < times["reach"]
                else times["grasp"] if t < times["grasp"] else last
            )
            for t in range(steps)
        ]
    else:
        raise ValueError(f"policy.goal: final, dense or stages, got {kind}")
    return goals[idx]


def demonstrations(cfg, env):
    """The recorded episodes the closed loop starts from: [{"video", "seed", "jpegs" (T,), "actions" (T - 1, 7),
    "commands" (T - 1, 7) OSC commands sent, "states" (T, 7) robot states, "cube" (T, 3)}]
    """
    e = cfg["episodes"]
    out = []
    for shard in shards(e["root"], e["split"]):
        for r in pq.read_table(
            shard, columns=["video", "frames", "actions"]
        ).to_pylist():
            if f"/{e['recorded_by']}/" not in r["video"]:
                continue
            folder = (
                "raw/" + r["video"].rsplit("/", 1)[0]
            )  # raw/<robot>/<task>/<recorded_by>/episode_<seed>
            raw = {
                name: pq.read_table(
                    hf_hub_download(
                        e["hf_repo"],
                        f"{folder}/{name}",
                        repo_type="dataset",
                        local_dir=e["root"],
                    )
                ).to_pylist()
                for name in ("sim.parquet", "state.parquet")
            }
            cube = np.array([s["object"] for s in raw["sim.parquet"]])
            if (
                e["lifted_only"]
                and not (
                    cube[:, 2] > env.table_height() + cfg["measures"]["lift"]
                ).any()
            ):
                continue
            out.append(
                {
                    "video": r["video"],
                    "seed": int(folder.rsplit("_", 1)[1]),
                    "jpegs": r["frames"],
                    "actions": np.array(r["actions"]),
                    "commands": [s["command"] for s in raw["sim.parquet"][:-1]],
                    "states": np.array([s["state"] for s in raw["state.parquet"]]),
                    "cube": cube,
                }
            )
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
    before the start, the first state repeated and a zero action. -> plan (H, 7), final mean cost of the elites
    """
    c, h = p["context"], p["horizon"]
    past_s = [states[0]] * (c - 1) + states
    past_a = [np.zeros(7)] * (c - 1) + actions
    with torch.no_grad():
        s = torch.stack(past_s[-c:])[None]  # (1, C, D)
        a = torch.tensor(
            np.array(past_a[len(past_a) - (c - 1) :]).reshape(-1, 7),
            dtype=torch.float32,
        )
        z = controller(a[None].to(s.device), s[:, :-1])  # (1, C - 1, z_dim)
        mean = controller.act_mean.expand(1, h, -1).clone()
        std = controller.act_std.expand(1, h, -1).clone()
        plan, costs, _ = cem(
            goal_cost(model, controller, s, z, goal[None]),
            mean,
            std,
            p["samples"],
            p["elites"],
            p["iterations"],
        )
    return plan[0].cpu().numpy(), costs[-1, 0].item()


def episode(env, cfg, demo, model=None, controller=None, device=None):
    """One closed-loop episode from the recording's seed -> per-step log and stage measures"""
    p, ex = cfg["policy"], cfg["executor"]
    torch.manual_seed(demo["seed"])
    rng = np.random.default_rng(
        demo["seed"]
    )  # as the collection: the scripted policy replays the recording
    obs = env.reset(demo["seed"])
    command = np.r_[np.zeros(6), -1.0]  # gripper open
    fps = env.cfg["control_freq"] / cfg["frameskip"]
    see = lambda: decode(
        [data_frame(env.frame(), fps, ex["video_crf"], ex["image_size"])]
    )

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
    log = {
        "stages": [stages(env, obs, cfg["measures"])],
        "distance": [float(np.linalg.norm(obs["robot0_eef_pos"] - obs["cube_pos"]))],
        "cube": [obs["cube_pos"].tolist()],
        "gripper": [float(command[6])],
        "plan": [],
        "cost": [],
        "time": [],
    }
    frames = [env.frame()] if cfg["render"]["video"] else None
    for t in range(cfg["steps"]):
        start = time.time()
        if p["type"] in COLLECTION_POLICIES:
            command = policy.send(obs) if t > 0 else command
        elif p["type"] == "recorded":
            command = to_command(
                demo["actions"][t], command, ex["K"], ex["gripper_threshold"]
            )
        elif p["type"] == "zero":
            command = to_command(np.zeros(7), command, ex["K"], ex["gripper_threshold"])
        elif p["type"] == "cem":
            plan, cost = plan_step(model, controller, states, achieved, goals[t], p)
            command = to_command(plan[0], command, ex["K"], ex["gripper_threshold"])
            log["plan"].append(plan[0].tolist())
            log["cost"].append(cost)
        else:
            raise ValueError(
                f"policy.type: random, scripted, recorded, zero or cem, got {p['type']}"
            )
        log["time"].append(time.time() - start)
        for _ in range(cfg["frameskip"]):
            obs = env.step(command)
        robot.append(env.robot_state(obs))
        if p["type"] == "cem":
            states.append(encode(model, see(), device)[0])
            achieved.append(
                states_to_actions(np.stack(robot[-2:]))[0]
            )  # the achieved action, as in the data
        log["stages"].append(stages(env, obs, cfg["measures"]))
        log["distance"].append(
            float(np.linalg.norm(obs["robot0_eef_pos"] - obs["cube_pos"]))
        )
        log["cube"].append(obs["cube_pos"].tolist())
        log["gripper"].append(float(command[6]))
        if frames is not None:
            frames.append(env.frame())

    first = {
        k: next((t for t, s in enumerate(log["stages"]) if s[k]), None)
        for k in ("reach", "grasp", "lift")
    }
    result = {
        "video": demo["video"],
        "seed": demo["seed"],
        **{k: v is not None for k, v in first.items()},
        "first": first,
        "grasped_steps": sum(s["grasp"] for s in log["stages"]),
        "closings": int((np.diff(log["gripper"]) > 0).sum()),
        "final_gripper_error": float(
            np.linalg.norm(robot[-1][:3] - demo["states"][-1][:3])
        ),
        "final_cube_error": float(
            np.linalg.norm(np.array(log["cube"][-1]) - demo["cube"][-1])
        ),
        "time_per_step": float(np.mean(log["time"])),
        "robot": np.array(robot).tolist(),
        **log,
    }
    return result, frames


def wilson(k, n, z=1.96):
    """95 % interval of a success rate k / n"""
    if n == 0:
        return [0.0, 1.0]
    p, d = k / n, 1 + z**2 / n
    centre, half = (p + z**2 / (2 * n)) / d, z * np.sqrt(
        p * (1 - p) / n + z**2 / (4 * n**2)
    ) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def summarize(results):
    """Per stage: count, rate and its interval, median step it is first met; medians of the other measures"""
    n = len(results)
    out = {"episodes": n}
    for k in ("reach", "grasp", "lift"):
        hits = [r["first"][k] for r in results if r[k]]
        out[k] = {
            "count": len(hits),
            "rate": len(hits) / n,
            "interval": wilson(len(hits), n),
            "median_first_step": float(np.median(hits)) if hits else None,
        }
    for k in (
        "grasped_steps",
        "closings",
        "final_gripper_error",
        "final_cube_error",
        "time_per_step",
    ):
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
                write_video(
                    frames,
                    out / "videos" / f"episode_{demo['seed']:06d}.mp4",
                    env.cfg["control_freq"] / cfg["frameskip"],
                    cfg["render"]["video_crf"],
                )
            results.append(result)
            f.write(json.dumps(result) + "\n")
            print(
                f"{i + 1} / {len(demos)} episode {demo['seed']}: reach {result['reach']}, grasp {result['grasp']}, "
                f"lift {result['lift']}, {result['time_per_step']:.2f} s per step",
                flush=True,
            )
    summary = summarize(results)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(f"{out}: {summary['episodes']} episodes")
    for k in ("reach", "grasp", "lift"):
        s = summary[k]
        print(
            f"  {k:6s} {s['count']:3d} [{100 * s['interval'][0]:.0f}-{100 * s['interval'][1]:.0f} %], "
            f"first at step {s['median_first_step']}"
        )
    print(
        f"  final error: gripper {100 * summary['final_gripper_error']:.1f} cm, cube {100 * summary['final_cube_error']:.1f} cm; "
        f"{summary['closings']:.0f} closings, {summary['time_per_step']:.2f} s per step"
    )
    return summary


# Offline battery (h2r plan offline): the CEM and its cost on windows of recorded episodes, no simulator. The measures
# and rules of the CEM rebuild (experimentation.ipynb, 7-9 Nov.), the same for every world model and its controller.


def episode_interval(score, episode, draws=10_000):
    """Mean of a per-window score and its 95 % interval, drawing episodes with replacement (the windows of one episode
    are not independent) -> [mean, low, high]"""
    score, episode = score.double().cpu(), episode.cpu()
    _, ep = episode.unique(return_inverse=True)
    n = int(ep.max()) + 1
    sums = torch.zeros(n, dtype=torch.float64).index_add_(0, ep, score)
    counts = torch.zeros(n, dtype=torch.float64).index_add_(
        0, ep, torch.ones_like(score)
    )
    idx = torch.randint(0, n, (draws, n), generator=torch.Generator().manual_seed(0))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    low, high = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return [score.mean().item(), low.item(), high.item()]


def dxyz(plan):
    """(..., H, 7) actions -> (..., 3) total end-effector move"""
    return plan[..., :3].sum(-2)


def recording_policies(root, split, data):
    """Policy that recorded each episode (scripted, random), read in the video path, or None (DROID); the order is the
    one of the encoded data, checked on the actions"""
    rows = [
        r
        for shard in shards(root, split)
        for r in pq.read_table(shard, columns=["video", "actions"]).to_pylist()
        if r["actions"] is not None
    ]
    assert len(rows) == len(data["a"])
    assert all(
        torch.allclose(torch.tensor(r["actions"], dtype=torch.float32), a.float())
        for r, a in zip(rows, data["a"])
    )
    policies = [
        next((p for p in ("scripted", "random") if f"/{p}/" in r["video"]), None)
        for r in rows
    ]
    return policies if any(policies) else None


def task_windows(data, length, context, policies):
    """Non-overlapping windows of length + 1 states (windows), float: s (N, length + 1, D), a (N, length, 7), the
    episode of each, and the groups the measures are read on: all windows, thirds of motion (distance from the current
    state to the goal, the last state) and each recording policy"""
    from h2r.data.dataset import windows

    w = windows(
        {
            "s": data["s"],
            "a": data["a"],
            "e": [torch.full((len(s),), i) for i, s in enumerate(data["s"])],
        },
        length,
    )
    s, a, episode = w["s"].float(), w["a"].float(), w["e"][:, 0]
    still = ((s[:, context - 1] - s[:, -1]) ** 2).mean(-1)
    order, n = still.argsort(), len(still)
    groups = {"all": torch.ones(n, dtype=torch.bool)}
    for name, part in zip(
        ("calm third", "middle third", "most moving third"),
        (order[: n // 3], order[n // 3 : 2 * n // 3], order[2 * n // 3 :]),
    ):
        groups[name] = torch.zeros(n, dtype=torch.bool)
        groups[name][part] = True
    for p in sorted(set(policies or [])):
        groups[p] = torch.tensor([policies[e] == p for e in episode.tolist()])
    return {"s": s, "a": a, "episode": episode, "still": still, "groups": groups}


def readings(score, w):
    """A per-window score -> {group: [mean, low, high]} with episode intervals"""
    return {
        name: episode_interval(score[mask], w["episode"][mask])
        for name, mask in w["groups"].items()
        if mask.any()
    }


def other_episodes(episode, k, seed=0):
    """(N, 1 + k): each window, then k windows drawn from other episodes"""
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(len(episode)):
        perm = torch.randperm(len(episode), generator=g)
        out.append(
            torch.cat([torch.tensor([i]), perm[episode[perm] != episode[i]][:k]])
        )
    return torch.stack(out)


@torch.no_grad()
def candidate_costs(
    model, controller, w, candidates, context, horizon, device, batch=128
):
    """Cost toward the last state of each window of the next `horizon` actions of the windows in candidates (the
    window's own true actions first, then other episodes' at the same place), and of the zero plan
    -> costs (N, 1 + k), zero (N,)"""
    costs, zero = [], []
    for idx in candidates.split(batch):
        s, a = w["s"][idx[:, 0]].to(device), w["a"][idx[:, 0]].to(device)
        c = s[:, :context]
        cost = goal_cost(
            model, controller, c, controller(a[:, : context - 1], c[:, :-1]), s[:, -1]
        )
        plans = w["a"][idx, context - 1 : context - 1 + horizon].to(device)
        costs.append(cost(plans).cpu())
        zero.append(cost(torch.zeros_like(plans[:, :1]))[:, 0].cpu())
    return torch.cat(costs), torch.cat(zero)


@torch.no_grad()
def imagined_goals(model, controller, w, context, actions, device, batch=128):
    """(N, D) the state the model imagines after the window's first `actions` true actions from its context"""
    goals = []
    for idx in torch.arange(len(w["s"])).split(batch):
        s, a = w["s"][idx].to(device), w["a"][idx].to(device)
        c = s[:, :context]
        z = controller(a[:, : context - 1], c[:, :-1])
        goals.append(
            model.rollout(
                c, z, a[:, context - 1 : context - 1 + actions][:, None], controller
            )[:, 0, -1].cpu()
        )
    return torch.cat(goals)


@torch.no_grad()
def plan_windows(model, controller, w, goals, context, horizon, p, device, batch=64):
    """CEM over `horizon` actions toward goals (N, D) from the context of each window, starting at the controller's
    action mean and std -> plans (N, horizon, 7), mean cost of the elites at the last iteration (N,)
    """
    plans, costs = [], []
    for idx in torch.arange(len(w["s"])).split(batch):
        s, a = w["s"][idx].to(device), w["a"][idx].to(device)
        c = s[:, :context]
        cost = goal_cost(
            model,
            controller,
            c,
            controller(a[:, : context - 1], c[:, :-1]),
            goals[idx].to(device),
        )
        mean = controller.act_mean.expand(len(idx), horizon, -1).clone()
        std = controller.act_std.expand(len(idx), horizon, -1).clone()
        plan, elite, _ = cem(
            cost, mean, std, p["samples"], p["elites"], p["iterations"]
        )
        plans.append(plan.cpu())
        costs.append(elite[-1].cpu())
        print(
            f"  CEM {sum(map(len, plans))} / {len(w['s'])} windows",
            end="\r",
            flush=True,
        )
    print()
    return torch.cat(plans), torch.cat(costs)


@torch.no_grad()
def sensitivity(model, controller, w, goals, context, horizon, device, batch=64):
    """Rise of the cost toward goals (the model's own rollout of the true actions, cost 0) when one action dimension
    is pushed by ± one training std on every step, mean of + and - -> (N, 7)"""
    out = []
    for idx in torch.arange(len(w["s"])).split(batch):
        s, a = w["s"][idx].to(device), w["a"][idx].to(device)
        c = s[:, :context]
        cost = goal_cost(
            model,
            controller,
            c,
            controller(a[:, : context - 1], c[:, :-1]),
            goals[idx].to(device),
        )
        true = a[:, context - 1 : context - 1 + horizon]
        push = torch.diag(controller.act_std)[None, :, None]  # (1, 7, 1, 7)
        out.append(
            cost(torch.cat([true[:, None] + push, true[:, None] - push], 1))
            .view(len(idx), 2, -1)
            .mean(1)
            .cpu()
        )
    return torch.cat(out)


def ridge_r2(x_train, y_train, x_val, y_val, lam=1.0):
    """Linear ridge fitted on train, R² of each target on val; float64 (float32 normal equations gave wrong R²)"""
    x_train, y_train, x_val, y_val = (
        x_train.double(),
        y_train.double(),
        x_val.double(),
        y_val.double(),
    )
    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    xt = torch.cat(
        [(x_train - mean) / std, torch.ones(len(x_train), 1, dtype=torch.float64)], 1
    )
    xv = torch.cat(
        [(x_val - mean) / std, torch.ones(len(x_val), 1, dtype=torch.float64)], 1
    )
    weights = torch.linalg.solve(
        xt.T @ xt + lam * torch.eye(xt.shape[1], dtype=torch.float64), xt.T @ y_train
    )
    return 1 - ((xv @ weights - y_val) ** 2).sum(0) / (
        (y_val - y_val.mean(0)) ** 2
    ).sum(0)


def state_probe(train, data):
    """Does the state hold the robot's motion? Linear probe of the change of x, y, z and gripper closure since the
    episode start (sum of the actions) from s_t - s_0, fitted on train, R² on data; control: the time step alone
    """

    def features(d):
        x = torch.cat([s.float() - s[:1].float() for s in d["s"]])
        y = torch.cat(
            [
                torch.cat([torch.zeros(1, 4), a.float()[:, [0, 1, 2, 6]].cumsum(0)])
                for a in d["a"]
            ]
        )
        t = torch.cat([torch.arange(len(s)) for s in d["s"]])
        return x, y, t

    (x_tr, y_tr, t_tr), (x_va, y_va, t_va) = features(train), features(data)
    n = int(max(t_tr.max(), t_va.max())) + 1
    time_tr, time_va = (torch.nn.functional.one_hot(t, n).float() for t in (t_tr, t_va))
    inputs = {
        "time only": (time_tr, time_va),
        "s_t - s_0": (x_tr, x_va),
        "s_t - s_0 + time": (
            torch.cat([x_tr, time_tr], 1),
            torch.cat([x_va, time_va], 1),
        ),
    }
    return {
        name: dict(
            zip(("x", "y", "z", "gripper"), ridge_r2(tr, y_tr, va, y_va).tolist())
        )
        for name, (tr, va) in inputs.items()
    }


def battery(model, controller, data, train, policies, cfg, device):
    """Every offline measure on one split -> (metrics, per-window values)"""
    c, h, g, k, p, rules = (
        cfg["context"],
        cfg["horizon"],
        cfg["goal"],
        cfg["candidates"],
        cfg["cem"],
        cfg["rules"],
    )
    out, saved = {"state_probe": state_probe(train, data)}, {}

    # 1. Cost gates: do the true actions cost less than other episodes' and than the zero plan?
    for gate, length in (("gate_near", c - 1 + h), ("gate_far", c - 1 + g)):
        w = task_windows(data, length, c, policies)
        costs, zero = candidate_costs(
            model, controller, w, other_episodes(w["episode"], k), c, h, device
        )
        rank = (costs[:, 1:] < costs[:, :1]).sum(
            1
        )  # candidates cheaper than the true actions
        out[gate] = {
            "windows": len(rank),
            "episodes": int(w["episode"].unique().numel()),
            "beats_other_episodes": readings(1 - rank / k, w),
            "beats_zero_plan": readings((costs[:, 0] < zero).float(), w),
            "hit_top1": readings((rank < 1).float(), w),
            "hit_top5": readings((rank < 5).float(), w),
            "chance_top1": 1 / (k + 1),
            "chance_top5": 5 / (k + 1),
            "true_cost": costs[:, 0].mean().item(),
            "still_cost": w["still"].mean().item(),
        }
        saved[gate] = {
            "episode": w["episode"],
            "rank": rank,
            "true_cost": costs[:, 0],
            "zero_cost": zero,
        }
    out["gate_near"][
        "rule"
    ] = f"beats_other_episodes all, lower bound >= {rules['near']}"
    out["gate_near"]["passes"] = (
        out["gate_near"]["beats_other_episodes"]["all"][1] >= rules["near"]
    )
    out["gate_far"]["rule"] = f"beats_zero_plan all, lower bound > {rules['far']}"
    out["gate_far"]["passes"] = (
        out["gate_far"]["beats_zero_plan"]["all"][1] > rules["far"]
    )

    # 2. Known case: the goal is the model's own rollout of the h true actions, so the true actions cost 0
    w = task_windows(data, c - 1 + g, c, policies)
    true = w["a"][:, c - 1 : c - 1 + h]
    moved = dxyz(true)
    goals = imagined_goals(model, controller, w, c, h, device)
    torch.manual_seed(0)
    plans, elite = plan_windows(model, controller, w, goals, c, h, p, device)
    push = sensitivity(model, controller, w, goals, c, h, device)
    out["known_case"] = {
        "cosine_dxyz": readings(
            torch.nn.functional.cosine_similarity(dxyz(plans), moved, dim=-1), w
        ),
        "amplitude_median": (dxyz(plans).norm(dim=-1) / moved.norm(dim=-1))
        .median()
        .item(),
        "final_elite_cost_median": elite.median().item(),
        "sensitivity_median": dict(
            zip(
                ("x", "y", "z", "rx", "ry", "rz", "gripper"),
                push.median(0).values.tolist(),
            )
        ),
    }
    saved["known_case"] = {"plans": plans, "elite_cost": elite, "sensitivity": push}

    # 3. Verdict: CEM toward the real goal, g steps after the current state; L1 error of the planned Δxyz
    torch.manual_seed(0)
    plans, elite = plan_windows(model, controller, w, w["s"][:, -1], c, h, p, device)
    other = w["a"][other_episodes(w["episode"], 1)[:, 1], c - 1 : c - 1 + h]
    errors = {
        "cem": (dxyz(plans) - moved).abs().sum(-1),
        "zero_plan": moved.abs().sum(-1),
        "other_episode": (dxyz(other) - moved).abs().sum(-1),
    }
    gain = errors["zero_plan"] - errors["cem"]
    whole = w["a"][:, c - 1 :, :3].sum(1)  # the true move up to the goal
    out["verdict"] = {
        "windows": len(gain),
        "dxyz_error": {name: readings(e, w) for name, e in errors.items()},
        "gain_on_zero_plan": readings(gain, w),
        "cosine_dxyz": readings(
            torch.nn.functional.cosine_similarity(dxyz(plans), moved, dim=-1), w
        ),
        "cosine_dxyz_to_goal": readings(
            torch.nn.functional.cosine_similarity(dxyz(plans), whole, dim=-1), w
        ),
        "amplitude_median": (dxyz(plans).norm(dim=-1) / moved.norm(dim=-1))
        .median()
        .item(),
        "rule": "gain_on_zero_plan lower bound > 0, all and most moving third",
    }
    out["verdict"]["passes"] = all(
        out["verdict"]["gain_on_zero_plan"][part][1] > 0
        for part in ("all", "most moving third")
    )
    saved["verdict"] = {
        "episode": w["episode"],
        "plans": plans,
        "elite_cost": elite,
        **errors,
    }
    return out, saved


def report(split, m):
    """The main numbers of a split, printed"""
    f = lambda r: f"{r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]"
    print(f"\n== {split}")
    for name, r2 in m["state_probe"].items():
        print(
            f"  state probe, {name:17s} R² "
            + "  ".join(f"{t} {v:+.2f}" for t, v in r2.items())
        )
    for gate in ("gate_near", "gate_far"):
        g = m[gate]
        print(
            f"  {gate} ({g['windows']} windows, {g['episodes']} episodes): vs other episodes "
            f"{f(g['beats_other_episodes']['all'])}, vs zero plan {f(g['beats_zero_plan']['all'])}, "
            f"hit top-1 {f(g['hit_top1']['all'])} top-5 {f(g['hit_top5']['all'])} "
            f"(chance {g['chance_top1']:.3f} / {g['chance_top5']:.3f}); {'PASSES' if g['passes'] else 'fails'} ({g['rule']})"
        )
    kc = m["known_case"]
    print(
        f"  known case: cosine Δxyz {f(kc['cosine_dxyz']['all'])}, amplitude {kc['amplitude_median']:.2f}, "
        f"elite cost {kc['final_elite_cost_median']:.5f}; sensitivity "
        + " ".join(f"{d} {v:.4f}" for d, v in kc["sensitivity_median"].items())
    )
    v = m["verdict"]
    print(
        f"  verdict ({v['windows']} windows): Δxyz L1 error "
        + ", ".join(f"{k} {f(r['all'])}" for k, r in v["dxyz_error"].items())
    )
    for part, r in v["gain_on_zero_plan"].items():
        print(
            f"    gain on the zero plan, {part:18s} {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]"
        )
    print(
        f"    cosine {f(v['cosine_dxyz']['all'])}, to the goal {f(v['cosine_dxyz_to_goal']['all'])}, "
        f"amplitude {v['amplitude_median']:.2f}; {'PASSES' if v['passes'] else 'FAILS'} ({v['rule']})"
    )


def offline(cfg):
    """h2r plan offline: the offline battery of a controller and its world model on recorded episodes; run folder with
    the config, metrics.json (every split) and windows-<split>.pt (the per-window values)
    """
    from h2r.models.controller import load_controller
    from h2r.training.controller import encoded
    from h2r.training.utils import make_autocast, pick_device

    device = pick_device()
    run = Path(cfg["controller"])
    ctrl_cfg = yaml.safe_load(open(run / "config.yaml"))
    ctrl_cfg["data"]["cache"] = cfg["cache"]
    model, controller = load_controller(run, device)
    model.history_size = cfg["context"]
    autocast = make_autocast(ctrl_cfg["precision"], device)
    out = Path(cfg["output_dir"]) / (
        cfg["name"] or f"{run.parent.name}-{run.name}-{Path(cfg['root']).name}"
    )
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"), sort_keys=False)

    train = encoded(model, ctrl_cfg, cfg["root"], "train", device, autocast)
    metrics = {}
    for split in cfg["splits"]:
        start = time.time()
        data = encoded(model, ctrl_cfg, cfg["root"], split, device, autocast)
        metrics[split], saved = battery(
            model,
            controller,
            data,
            train,
            recording_policies(cfg["root"], split, data),
            cfg,
            device,
        )
        metrics[split]["seconds"] = time.time() - start
        torch.save(saved, out / f"windows-{split}.pt")
        json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
        report(split, metrics[split])
    print(f"\n{out}")
    return metrics
