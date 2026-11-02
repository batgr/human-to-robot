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
    ACTION_NAMES,
    PROBED,
    PROBED_NAMES,
    REGIMES,
    ControlledRollout,
    PlanningError,
    action_classes,
    action_probe,
    class_probes,
    counterfactual,
    fig_actions,
    fig_cem,
    fig_counterfactual,
    fig_latent_actions,
    fig_regimes,
    fig_spectra,
    fig_state_probe,
    fig_trajectories,
    mean_r2,
    pca_basis,
    plan,
    r2,
    regimes,
    ridge,
    spectrum,
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
    fp16, a: [(T_e - 1, act_dim)] the real actions, and x: [(T_e, state_dim)] the absolute states when the shards have
    them (the physical probe)
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
    out = {"s": states, "a": [torch.tensor(ep["actions"], dtype=torch.float32) for ep in episodes]}
    if episodes and all(ep.get("states") is not None for ep in episodes):
        out["x"] = [torch.tensor(ep["states"], dtype=torch.float32) for ep in episodes]
    return out


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
    error, trace = PlanningError(), []
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
                trace=trace,
            )
        error.update(planned.float().cpu(), actions[i : i + chunk])
    out = error.compute() | {"clips": len(clips)}

    # per CEM iteration, over every clip: the trace of each chunk, one entry per iteration
    its = p["iterations"]
    per_it = [[trace[c * its + k] for c in range(len(trace) // its)] for k in range(its)]
    curve = {"clipped": [], "best_cost": [], "dxyz": [], "dxyz_still": out["dxyz_still"]}
    for k, entries in enumerate(per_it):
        curve["clipped"].append(torch.cat([e["clipped"] for e in entries]).mean().item())
        curve["best_cost"].append(torch.cat([e["best_cost"] for e in entries]).float().mean().item())
        it_error = PlanningError()
        it_error.update(torch.cat([e["best"] for e in entries]), actions)
        curve["dxyz"].append(it_error.compute()["dxyz"])
    dims = torch.stack([torch.stack([e["clipped_dims"] for e in entries]).mean(0) for entries in per_it]).mean(0)
    names = ACTION_NAMES if len(dims) == len(ACTION_NAMES) else [str(i) for i in range(len(dims))]
    out |= {"clipped_first": curve["clipped"][0], "clipped_last": curve["clipped"][-1],
            "dxyz_first": curve["dxyz"][0]}
    out |= {f"clipped_{n}": float(v) for n, v in zip(names, dims)}
    return out, curve


@torch.no_grad()
def analysis(model, ctrl, train_set, test_set, mean_z, cfg, device, autocast):
    """The test analysis, metrics and figures:
    - test_classes/: do the latent actions name the real action (diagnostics E11 A)?
    - test_counterfactual/: what does each action dimension do through the controller (E11 B, when it sees the action)?
    - test_regimes/: open-loop MSE per regime of the test windows (still, free motion, gripper) and horizon;
    - test_state_probe/: the end-effector state (x, y, z, gripper) decoded linearly from the states, fitted on real
      training states: r2 on real test states first (what the representation holds: below it, the rollouts cannot be
      blamed), then on the states predicted h steps ahead by each source and on the persistence s_0 (needs the absolute
      states in the shards);
    - test_spectra/: effective rank and axes for 90 % of the variance of the states, of the real changes of state and of
      the imagined ones (controller, first step): the motion can use few directions in states of high rank;
    - figures: actions, latent_actions, counterfactual, regimes, state_probe, spectra, trajectories"""
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
    bounds = cfg["planning"].get("bounds")
    figs = {
        "actions": fig_actions(a_train, None if bounds is None else torch.tensor(bounds)),
        "latent_actions": fig_latent_actions(feats["test"], labels["test"], conf),
    }

    if "action" in ctrl.inputs:
        spread = (mu_train - mean_z.cpu()).norm(dim=-1).median().item()
        step = ds_train.float().flatten(1).norm(dim=-1).median().item()
        with autocast:
            out, cf = counterfactual(model, ctrl, transitions(test_set)[0], step, spread, gen,
                                     n=c["counterfactual_states"])
        metrics |= {f"test_counterfactual/{k}": v for k, v in out.items()}
        figs["counterfactual"] = fig_counterfactual(*cf)

    # the physical probe, fitted on real training states
    probe = None
    if "x" in train_set and "x" in test_set:
        s_all, x_all = torch.cat(train_set["s"]), torch.cat(train_set["x"])[:, PROBED]
        idx = torch.randperm(len(s_all), generator=gen)[: c["max_points"]]
        probe = ridge(s_all[idx].float().flatten(1), x_all[idx])
        real = r2(probe(torch.cat(test_set["s"]).float().flatten(1)), torch.cat(test_set["x"])[:, PROBED])
        metrics |= {f"test_state_probe/real_r2_{n}": v for n, v in zip(PROBED_NAMES, real)}
        metrics["test_state_probe/real_r2"] = mean_r2(real)

    # the test windows: rollouts by regime, the probe on the predicted states, the imagined motion
    win = windows(test_set, cfg["horizon"])
    groups = regimes(win["a"], c["still_xyz"], c["gripper_change"])
    control = ControlledRollout(mean_z)
    decoded = {k: [] for k in ["ctrl", "idm", "mean", "persistence"]}
    ds_real, ds_imagined = [], []
    for i in range(0, len(win["a"]), cfg["batch_size"]):
        batch = {k: v[i : i + cfg["batch_size"]].to(device).float() for k, v in win.items()}
        with autocast:
            preds = control.update(model, ctrl, batch["s"], batch["mu"], batch["a"], groups[i : i + cfg["batch_size"]])
        s0 = batch["s"][:, 0].flatten(1)
        ds_real.append((batch["s"][:, 1].flatten(1) - s0).cpu())
        ds_imagined.append((preds["ctrl"][:, 0].float().flatten(1) - s0).cpu())
        if probe is not None:
            b, h = batch["a"].shape[:2]
            for k in ["ctrl", "idm", "mean"]:
                decoded[k].append(probe(preds[k].float().flatten(2).flatten(0, 1).cpu()).view(b, h, -1))
            decoded["persistence"].append(probe(s0.cpu()).view(b, 1, -1).expand(-1, h, -1))
    rollout = control.compute()
    metrics |= {f"test_regimes/{k}": v for k, v in rollout.items() if k.startswith(tuple(f"{r}_" for r in REGIMES))}
    figs["regimes"] = fig_regimes(rollout, cfg["horizon"])

    if probe is not None:
        true = win["x"][:, 1:][..., PROBED]  # (W, H, 4)
        curves = {"real": metrics["test_state_probe/real_r2"]}
        for k, chunks in decoded.items():
            pred = torch.cat(chunks)
            curves[k] = [mean_r2(r2(pred[:, h], true[:, h])) for h in range(true.size(1))]
            metrics |= {f"test_state_probe/{k}_r2_h{h + 1}": v for h, v in enumerate(curves[k])}
        figs["state_probe"] = fig_state_probe(curves)

    spectra = {
        "states": spectrum(s_train[torch.randperm(len(s_train), generator=gen)[: c["max_points"]]].float().flatten(1)),
        "real changes Δs": spectrum(torch.cat(ds_real)),
        "imagined changes (controller, h = 1)": spectrum(torch.cat(ds_imagined)),
    }
    for name, key in zip(spectra, ["s", "ds_real", "ds_imagined"]):
        metrics |= {f"test_spectra/{key}_{k}": v for k, v in spectra[name][1].items()}
    figs["spectra"] = fig_spectra(spectra)

    # a few test windows in the PCA of the real training states
    mean, axes = pca_basis(s_train[torch.randperm(len(s_train), generator=gen)[: c["max_points"]]].float().flatten(1))
    pick = torch.randperm(len(win["a"]), generator=gen)[: c["trajectories"]]
    s, mu, a = (win[k][pick].to(device).float() for k in ["s", "mu", "a"])
    with autocast:
        ctrl_pred = model.rollout(s[:, :1], mu[:, :0], a[:, None], ctrl)[:, 0].float()
        idm_pred = model.rollout(s[:, :1], mu[:, :0], mu[:, None])[:, 0].float()

    def project(traj):
        return ((traj.flatten(2).cpu() - mean) @ axes).numpy()

    background = ((s_train[torch.randperm(len(s_train), generator=gen)[:5000]].float().flatten(1) - mean) @ axes)
    figs["trajectories"] = fig_trajectories(
        background.numpy(), project(s), project(torch.cat([s[:, :1], ctrl_pred], 1)),
        project(torch.cat([s[:, :1], idm_pred], 1)),
    )
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
    more, figs = analysis(model, ctrl, train_set, test_set, mean_z, cfg, device, autocast)
    if "action" in ctrl.inputs:  # planning over actions the controller does not see means nothing
        out, curve = planning(model, ctrl, test_set, cfg, device, autocast)
        metrics |= {f"test_planning/{k}": v for k, v in out.items()}
        figs["cem"] = fig_cem(curve)
    log(run_dir, cfg["iterations"], metrics | more, wandb_run)
    save_figures(figs, run_dir, cfg["iterations"], wandb_run)
    if wandb_run is not None:
        wandb_run.finish()
