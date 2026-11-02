"""Train the controller (real action + current state -> the IDM's latent action) for a frozen world model, on episodes
with actions, and measure on the test episodes (h2r.metrics.control): how much of the world model's open-loop skill the
real actions keep, whether z encodes the robot's motion, and offline planning to a goal frame
"""

import math
import time
from pathlib import Path

import torch

from h2r.data.dataset import decode, download, read_episodes, windows
from h2r.metrics.control import ControlledRollout, PlanningError, action_probe, plan
from h2r.models.controller import build_controller
from h2r.models.jepa import load_world_model
from h2r.training.utils import (
    init_wandb,
    log,
    make_autocast,
    make_run_dir,
    pick_device,
    save_checkpoint,
    seed_everything,
    warmup_cosine,
)


@torch.no_grad()
def encode_episodes(model, episodes, device, autocast, chunk=64):
    """Episodes of T frames each (h2r data collect: steps + 1) -> s: (E, T, ...) the world model's states in fp16,
    a: (E, T - 1, act_dim) the real actions
    """
    lengths = {len(ep["frames"]) for ep in episodes}
    assert len(lengths) == 1, f"episodes of different lengths: {sorted(lengths)}"
    assert all(
        len(ep["actions"]) == len(ep["frames"]) - 1 for ep in episodes
    ), "one action per transition"
    states = []
    for ep in episodes:
        px = decode(ep["frames"])
        with autocast:
            emb = torch.cat(
                [
                    model.encode(px[i : i + chunk][None].to(device))[0]
                    for i in range(0, len(px), chunk)
                ]
            )
        states.append(emb.half().cpu())
    actions = torch.tensor([ep["actions"] for ep in episodes], dtype=torch.float32)
    return {"s": torch.stack(states), "a": actions}


def encoded(model, cfg, split, device, autocast):
    """States and actions of a split, encoded once per world model run and dataset"""
    data = cfg["data"]
    path = (
        Path(data["cache"])
        / Path(cfg["world_model"]).name
        / f"{Path(data['root']).name}-{split}.pt"
    )
    if not path.exists():
        print(f"encoding the {split} episodes with {cfg['world_model']}")
        out = encode_episodes(
            model, read_episodes(data["root"], split), device, autocast
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, path)
    return torch.load(path)


def transitions(data):
    """Every transition of every episode: s (M, ...), a (M, act_dim), mu (M, z_dim)"""
    return (
        data["s"][:, :-1].flatten(0, 1),
        data["a"].flatten(0, 1),
        data["mu"].flatten(0, 1),
    )


def controller_loss(ctrl, s, a, mu):
    """L2 to the IDM's latent action, summed over its dimensions (Garrido et al. 2026); s: (B, D) or (B, N, D)"""
    return (ctrl(a, s.flatten(1).float()).float() - mu).pow(2).sum(-1).mean()


@torch.no_grad()
def evaluate(model, ctrl, data, mean_z, cfg, device, autocast, split):
    """{split}/: loss of z on every transition, loss_mean_z of the trivial controller (always the mean training mu, the
    total variance of mu) and r2 = 1 - loss / loss_mean_z;
    {split}_rollout/: ControlledRollout on the windows of the episodes"""
    ctrl.eval()
    s, a, mu = transitions(data)
    loss, chunk = 0.0, 4096
    for i in range(0, len(a), chunk):
        batch = (x[i : i + chunk].to(device) for x in (s, a, mu))
        with autocast:
            loss += controller_loss(ctrl, *batch).item() * len(a[i : i + chunk])
    loss /= len(a)
    baseline = (mu - mean_z.cpu()).pow(2).sum(-1).mean().item()

    control = ControlledRollout(mean_z)
    win = windows(data, cfg["horizon"])
    for i in range(0, len(win["a"]), cfg["batch_size"]):
        batch = {
            k: v[i : i + cfg["batch_size"]].to(device).float() for k, v in win.items()
        }
        with autocast:
            control.update(model, ctrl, batch["s"], batch["mu"], batch["a"])
    ctrl.train()
    return {f"{split}/loss": loss, f"{split}/loss_mean_z": baseline, f"{split}/r2": 1 - loss / baseline} | {
        f"{split}_rollout/{k}": v for k, v in control.compute().items()
    }


@torch.no_grad()
def planning(model, ctrl, data, cfg, device, autocast, chunk=32):
    """PlanningError on one clip of horizon + 1 frames per episode, at a seeded random start, for a seeded draw of
    planning.episodes episodes (the same for every world model)"""
    p = cfg["planning"]
    gen = torch.Generator().manual_seed(cfg["seed"])
    T = data["s"].size(1)
    episodes = torch.randperm(len(data["s"]), generator=gen)[: p["episodes"]].tolist()
    E = len(episodes)
    t = torch.randint(T - p["horizon"], (E,), generator=gen)
    steps = t[:, None] + torch.arange(p["horizon"] + 1)
    states = torch.stack(
        [data["s"][e, steps[i]] for i, e in enumerate(episodes)]
    )  # (E, horizon + 1, ...)
    actions = torch.stack(
        [data["a"][e, steps[i, :-1]] for i, e in enumerate(episodes)]
    )  # (E, horizon, act_dim)
    error = PlanningError()
    ctrl.eval()
    for i in range(0, E, chunk):
        s = states[i : i + chunk].to(device).float()
        with autocast:
            planned = plan(
                model,
                ctrl,
                s[:, 0],
                s[:, -1],
                p["horizon"],
                p["samples"],
                p["elites"],
                p["iterations"],
                gen,
            )
        error.update(planned.float().cpu(), actions[i : i + chunk])
    return error.compute()


def train(cfg):
    if not cfg["world_model"]:
        raise ValueError(
            "world_model: the run of the world model, e.g. --set world_model=outputs/wm-..."
        )
    seed_everything(cfg["seed"])
    device = pick_device()
    autocast = make_autocast(cfg["precision"], device)

    download(cfg["data"]["hf_repo"], cfg["data"]["root"])
    model, wm_cfg = load_world_model(cfg["world_model"], device)
    splits = []
    for split in ("train", "val", "test"):
        data = encoded(model, cfg, split, device, autocast)
        with torch.no_grad(), autocast:  # the IDM's latent actions of every transition (eval: z = mu)
            data["mu"] = torch.cat(
                [model.idm(s.to(device).float())[0].float().cpu() for s in data["s"].split(256)]
            )
        splits.append(data)
    train_set, val_set, test_set = splits
    s, a, mu = (x.to(device) for x in transitions(train_set))
    mean_z = mu.mean(0)

    ctrl = build_controller(cfg["model"], wm_cfg).to(device)
    ctrl.set_action_stats(a)
    optim = cfg["optim"]
    opt = torch.optim.AdamW(
        ctrl.parameters(), lr=optim["lr"], weight_decay=optim["weight_decay"]
    )
    schedule = warmup_cosine(
        int(optim["warmup"] * cfg["iterations"]), cfg["iterations"]
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)

    run_dir = make_run_dir(cfg)
    wandb_run = init_wandb(cfg, run_dir)
    print(
        f"run: {run_dir} | device: {device} | world model: {cfg['world_model']} | "
        f"controller params: {sum(p.numel() for p in ctrl.parameters())} | train transitions: {len(a)}"
    )

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
            metrics = {
                "train/loss": total.item() / n,
                "optim/lr": sched.get_last_lr()[0],
                "system/step_s": (time.time() - t0) / n,
            }
            log(run_dir, step, metrics, wandb_run)
            total, t0 = 0.0, time.time()

        if step % cfg["eval_every"] == 0 or step == cfg["iterations"]:
            metrics = evaluate(
                model, ctrl, val_set, mean_z, cfg, device, autocast, "val"
            )
            log(run_dir, step, metrics, wandb_run)
            if metrics["val/loss"] < best:
                best = metrics["val/loss"]
                save_checkpoint(
                    run_dir / "best.pt",
                    model=ctrl.state_dict(),
                    step=step,
                    val_loss=best,
                )
            t0 = time.time()

    # the test split, once, with the best controller (selected on val)
    ctrl.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model"])
    metrics = evaluate(model, ctrl, test_set, mean_z, cfg, device, autocast, "test")
    _, a_test, mu_test = transitions(test_set)
    metrics |= {
        f"test_probe/{k}": v
        for k, v in action_probe(mu.cpu(), a.cpu(), mu_test, a_test).items()
    }
    metrics |= {
        f"test_planning/{k}": v
        for k, v in planning(model, ctrl, test_set, cfg, device, autocast).items()
    }
    log(run_dir, cfg["iterations"], metrics, wandb_run)
    if wandb_run is not None:
        wandb_run.finish()
