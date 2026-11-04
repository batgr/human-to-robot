"""Train the controller (real action + current state -> the IDM's latent action) for a frozen world model, on episodes
with actions, and measure it on the test episodes as Garrido et al. 2026 (h2r.metrics.control): the rollouts with the
controller against those with the IDM, and offline planning to a goal frame
"""

import math
import time
from pathlib import Path

import torch

from h2r.data.dataset import decode, download, read_episodes, windows
from h2r.metrics.control import ControlledRollout, PlanningError, plan
from h2r.models.controller import build_controller
from h2r.models.jepa import load_world_model
from h2r.training.utils import (
    build_optimizer,
    init_wandb,
    log,
    make_autocast,
    make_run_dir,
    pick_device,
    save_checkpoint,
    seed_everything,
)


@torch.no_grad()
def encode_episodes(model, episodes, device, autocast, chunk=64):
    """Episodes of T_e frames each (they may differ in length: DROID) -> s: [(T_e, ...)] the world model's states in
    fp16, a: [(T_e - 1, act_dim)] the real actions"""
    assert all(len(ep["actions"]) == len(ep["frames"]) - 1 for ep in episodes), "one action per transition"
    states = []
    for ep in episodes:
        px = decode(ep["frames"])
        with autocast:
            emb = torch.cat([model.encode(px[i : i + chunk][None].to(device))[0] for i in range(0, len(px), chunk)])
        states.append(emb.half().cpu())
    return {"s": states, "a": [torch.tensor(ep["actions"], dtype=torch.float32) for ep in episodes]}


def encoded(model, cfg, split, device, autocast):
    """States and actions of a split, encoded once per world model run and dataset (lists of episodes)"""
    data = cfg["data"]
    path = Path(data["cache"]) / Path(cfg["world_model"]).name / f"{Path(data['root']).name}-{split}.pt"
    if not path.exists():
        print(f"encoding the {split} episodes with {cfg['world_model']}")
        out = encode_episodes(model, read_episodes(data["root"], split), device, autocast)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, path)
    return {k: list(v) for k, v in torch.load(path).items() if k in ("s", "a")}


def transitions(data):
    """Every transition of every episode: s (M, ...) the state it starts from, a (M, act_dim), mu (M, z_dim)"""
    return torch.cat([s[:-1] for s in data["s"]]), torch.cat(data["a"]), torch.cat(data["mu"])


def controller_loss(ctrl, s, a, mu):
    """L2 to the IDM's latent action, summed over its dimensions (Garrido et al. 2026); s: (B, D) or (B, N, D)"""
    return (ctrl(a, s.flatten(1).float()).float() - mu).pow(2).sum(-1).mean()


@torch.no_grad()
def evaluate(model, ctrl, data, cfg, device, autocast, split):
    """{split}/loss: the controller's loss on every transition; {split}_rollout/: ControlledRollout on the windows of
    the episodes"""
    ctrl.eval()
    s, a, mu = transitions(data)
    loss, chunk = 0.0, 4096
    for i in range(0, len(a), chunk):
        with autocast:
            loss += controller_loss(ctrl, *(x[i : i + chunk].to(device) for x in (s, a, mu))).item() * len(a[i : i + chunk])
    control = ControlledRollout()
    win = windows(data, cfg["horizon"])
    for i in range(0, len(win["a"]), cfg["batch_size"]):
        batch = {k: v[i : i + cfg["batch_size"]].to(device).float() for k, v in win.items()}
        with autocast:
            control.update(model, ctrl, batch["s"], batch["mu"], batch["a"])
    ctrl.train()
    return {f"{split}/loss": loss / len(a)} | {f"{split}_rollout/{k}": v for k, v in control.compute().items()}


@torch.no_grad()
def planning(model, ctrl, data, cfg, device, autocast, chunk=16):
    """PlanningError, protocol of Garrido et al. 2026 (from Terver et al. 2025): planning.clips_per_video seeded random
    clips in each of a seeded draw of planning.videos test episodes (16 x 4 = 64), the same for every world model.
    A clip: the context states s_t-C+1..s_t (the latent actions linking them from the controller and the real
    actions), the goal s_t+horizon, the true actions a_t..a_t+horizon-1"""
    p = cfg["planning"]
    C, H = p["context"], p["horizon"]
    gen = torch.Generator().manual_seed(cfg["seed"])
    long_enough = [e for e, s in enumerate(data["s"]) if len(s) >= C + H]
    clips = []
    for e in (long_enough[i] for i in torch.randperm(len(long_enough), generator=gen)[: p["videos"]].tolist()):
        T = len(data["s"][e])
        clips += [(e, t) for t in torch.randint(C - 1, T - H, (p["clips_per_video"],), generator=gen).tolist()]
    context = torch.stack([data["s"][e][t - C + 1 : t + 1] for e, t in clips])  # (E, C, ...)
    goal = torch.stack([data["s"][e][t + H] for e, t in clips])
    past_a = torch.stack([data["a"][e][t - C + 1 : t] for e, t in clips])  # (E, C - 1, act_dim)
    actions = torch.stack([data["a"][e][t : t + H] for e, t in clips])  # (E, H, act_dim)
    bounds = None if p.get("bounds") is None else torch.tensor(p["bounds"], device=device)
    error = PlanningError()
    ctrl.eval()
    for i in range(0, len(clips), chunk):
        ctx = context[i : i + chunk].to(device).float()
        with autocast:
            past = ctrl(past_a[i : i + chunk].to(device), ctx[:, :-1].flatten(2))
            planned = plan(model, ctrl, ctx, past, goal[i : i + chunk].to(device).float(), H, p["samples"],
                           p["elites"], p["iterations"], gen, init_std=p.get("init_std", 1.0), bounds=bounds)
        error.update(planned.float().cpu(), actions[i : i + chunk])
    return error.compute() | {"clips": len(clips)}


def train(cfg):
    if not cfg["world_model"]:
        raise ValueError("world_model: the run of the world model, e.g. --set world_model=outputs/wm-...")
    seed_everything(cfg["seed"])
    device = pick_device()
    autocast = make_autocast(cfg["precision"], device)

    download(cfg["data"]["hf_repo"], cfg["data"]["root"])
    model, wm_cfg = load_world_model(cfg["world_model"], device)
    splits = []
    for split in ("train", "val", "test"):
        data = encoded(model, cfg, split, device, autocast)
        with torch.no_grad(), autocast:  # the IDM's latent actions of every transition (eval: z = mu)
            data["mu"] = [model.idm(s[None].to(device).float())[0][0].float().cpu() for s in data["s"]]
        splits.append(data)
    train_set, val_set, test_set = splits
    s, a, mu = (x.to(device) for x in transitions(train_set))

    ctrl = build_controller(cfg["model"], wm_cfg).to(device)
    ctrl.set_action_stats(a)
    opt, sched = build_optimizer(ctrl, cfg["optim"], cfg["iterations"])

    run_dir = make_run_dir(cfg)
    wandb_run = init_wandb(cfg, run_dir)
    print(f"run: {run_dir} | device: {device} | world model: {cfg['world_model']} | "
          f"controller params: {sum(p.numel() for p in ctrl.parameters())} | train transitions: {len(a)}")

    gen = torch.Generator().manual_seed(cfg["seed"])
    best, total, t0 = math.inf, 0.0, time.time()
    for step in range(1, cfg["iterations"] + 1):
        idx = torch.randint(len(a), (cfg["batch_size"],), generator=gen).to(device)
        with autocast:
            loss = controller_loss(ctrl, s[idx], a[idx], mu[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        total += loss.detach()

        if step % cfg["log_every"] == 0:
            n = cfg["log_every"]
            metrics = {"train/loss": total.item() / n, "optim/lr": sched.get_last_lr()[0],
                       "system/step_s": (time.time() - t0) / n}
            log(run_dir, step, metrics, wandb_run)
            total, t0 = 0.0, time.time()

        if step % cfg["eval_every"] == 0 or step == cfg["iterations"]:
            metrics = evaluate(model, ctrl, val_set, cfg, device, autocast, "val")
            log(run_dir, step, metrics, wandb_run)
            if metrics["val/loss"] < best:
                best = metrics["val/loss"]
                save_checkpoint(run_dir / "best.pt", model=ctrl.state_dict(), step=step, val_loss=best)
            t0 = time.time()

    # the test split, once, with the best controller (selected on val)
    ctrl.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model"])
    ctrl.eval()
    metrics = evaluate(model, ctrl, test_set, cfg, device, autocast, "test")
    if "action" in ctrl.inputs:  # planning over actions the controller does not see means nothing
        metrics |= {f"test_planning/{k}": v for k, v in planning(model, ctrl, test_set, cfg, device, autocast).items()}
    log(run_dir, cfg["iterations"], metrics, wandb_run)
    if wandb_run is not None:
        wandb_run.finish()
