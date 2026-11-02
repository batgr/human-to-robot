"""Train the controller (real action + current state -> the IDM's latent action) for a frozen world model, on episodes
with actions, and measure on the test episodes (h2r.metrics.control): how much of the world model's open-loop skill the
real actions keep, whether z encodes the robot's motion, and offline planning to a goal frame
"""

import math
import time
from pathlib import Path

import torch

from h2r.data.dataset import decode, download, read_episodes, windows
from h2r.metrics.control import (
    ControlledRollout,
    PlanningError,
    action_classes,
    action_probe,
    class_probes,
    counterfactual,
    figures,
    plan,
)
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
    """Episodes of T_e frames each (they may differ in length: DROID) -> s: [(T_e, ...)] the world model's states in
    fp16, a: [(T_e - 1, act_dim)] the real actions
    """
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
    actions = [torch.tensor(ep["actions"], dtype=torch.float32) for ep in episodes]
    return {"s": states, "a": actions}


def encoded(model, cfg, split, device, autocast):
    """States and actions of a split, encoded once per world model run and dataset (lists of episodes; caches of
    stacked episodes from before are read the same way)"""
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
    return {k: list(v) for k, v in torch.load(path).items()}


def transitions(data):
    """Every transition of every episode: s (M, ...) the state it starts from, a (M, act_dim), mu (M, z_dim),
    ds (M, ...) the change of state s_t+1 - s_t"""
    return (
        torch.cat([s[:-1] for s in data["s"]]),
        torch.cat(data["a"]),
        torch.cat(data["mu"]),
        torch.cat([(s[1:].float() - s[:-1].float()).half() for s in data["s"]]),
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
    s, a, mu, _ = transitions(data)
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
def planning(model, ctrl, data, cfg, device, autocast, chunk=16):
    """PlanningError, protocol of Garrido et al. 2026 (from Terver et al. 2025): planning.clips_per_video seeded random
    clips in each of a seeded draw of planning.videos test episodes (16 x 4 = 64), the same for every world model.
    A clip: the context states s_t-C+1..s_t (the latent actions linking them from the controller and the real
    actions), the goal s_t+horizon, the true actions a_t..a_t+horizon-1"""
    p = cfg["planning"]
    C, H = p["context"], p["horizon"]
    gen = torch.Generator().manual_seed(cfg["seed"])
    long_enough = [e for e, s in enumerate(data["s"]) if len(s) >= C + H]
    pick = torch.randperm(len(long_enough), generator=gen)[: p["videos"]].tolist()
    clips = []
    for e in (long_enough[i] for i in pick):
        T = len(data["s"][e])
        for t in torch.randint(C - 1, T - H, (p["clips_per_video"],), generator=gen).tolist():
            clips.append((e, t))
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
            planned = plan(
                model,
                ctrl,
                ctx,
                past,
                goal[i : i + chunk].to(device).float(),
                H,
                p["samples"],
                p["elites"],
                p["iterations"],
                gen,
                init_std=p.get("init_std", 1.0),
                bounds=bounds,
            )
        error.update(planned.float().cpu(), actions[i : i + chunk])
    return error.compute() | {"clips": len(clips)}


@torch.no_grad()
def analysis(model, ctrl, train_set, test_set, mean_z, cfg, device, autocast):
    """Do the latent actions name the real action (diagnostics E11 A), and what does each action dimension do through
    the controller (E11 B, only when it sees the action)? Metrics, and the figures (h2r.metrics.control.figures)"""
    c = cfg["analysis"]
    gen = torch.Generator().manual_seed(cfg["seed"])
    feats, labels = {}, {}
    s_train, a_train, mu_train, ds_train = transitions(train_set)
    std = a_train.std(0)
    for name, (s, a, mu, ds) in [("train", (s_train, a_train, mu_train, ds_train)), ("test", transitions(test_set))]:
        idx = torch.randperm(len(a), generator=gen)[: c["max_points"]]
        s, a, mu, ds = s[idx], a[idx], mu[idx], ds[idx]
        with autocast:
            z = torch.cat([ctrl(x.to(device), y.to(device).float().flatten(1)).float().cpu()
                           for x, y in zip(a.split(4096), s.split(4096))])
        feats[name] = {"mu": mu, "controller_z": z, "s_t": s.float().flatten(1), "ds": ds.float().flatten(1)}
        labels[name] = action_classes(a, std, c["gripper_change"])
    out, conf = class_probes(feats["train"], feats["test"], labels["train"], labels["test"])
    metrics = {f"test_classes/{k}": v for k, v in out.items()}

    cf = None
    if "action" in ctrl.inputs:
        spread = (mu_train - mean_z.cpu()).norm(dim=-1).median().item()
        step = ds_train.float().flatten(1).norm(dim=-1).median().item()
        s_test = transitions(test_set)[0]
        with autocast:
            out, cf = counterfactual(model, ctrl, s_test, step, spread, gen, n=c["counterfactual_states"])
        metrics |= {f"test_counterfactual/{k}": v for k, v in out.items()}
    bounds = cfg["planning"].get("bounds")
    figs = figures(a_train, None if bounds is None else torch.tensor(bounds), feats["test"], labels["test"], conf, cf)
    return metrics, figs


def save_figures(figs, run_dir, step, wandb_run):
    """PNG files in the run directory, and W&B images under test_figures/"""
    from matplotlib import pyplot as plt

    for name, fig in figs.items():
        fig.savefig(run_dir / f"{name}.png", dpi=120)
    if wandb_run is not None:
        import wandb

        wandb_run.log({f"test_figures/{k}": wandb.Image(f) for k, f in figs.items()}, step=step)
    for fig in figs.values():
        plt.close(fig)


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
            data["mu"] = [model.idm(s[None].to(device).float())[0][0].float().cpu() for s in data["s"]]
        splits.append(data)
    train_set, val_set, test_set = splits
    s, a, mu = (x.to(device) for x in transitions(train_set)[:3])
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
    ctrl.eval()
    metrics = evaluate(model, ctrl, test_set, mean_z, cfg, device, autocast, "test")
    _, a_test, mu_test, _ = transitions(test_set)
    metrics |= {
        f"test_probe/{k}": v
        for k, v in action_probe(mu.cpu(), a.cpu(), mu_test, a_test).items()
    }
    if "action" in ctrl.inputs:  # planning over actions the controller does not see means nothing
        metrics |= {
            f"test_planning/{k}": v
            for k, v in planning(model, ctrl, test_set, cfg, device, autocast).items()
        }
    more, figs = analysis(model, ctrl, train_set, test_set, mean_z, cfg, device, autocast)
    log(run_dir, cfg["iterations"], metrics | more, wandb_run)
    save_figures(figs, run_dir, cfg["iterations"], wandb_run)
    if wandb_run is not None:
        wandb_run.finish()
