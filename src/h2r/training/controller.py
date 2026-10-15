"""Train the controller (real action + current frame -> latent action) on episodes with actions."""

import io
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image

from h2r.models.controller import Controller
from h2r.training.utils import (
    download_data,
    init_wandb,
    load_checkpoint,
    make_logger,
    new_metadata,
    prepare_run,
    resumed_metadata,
    save_checkpoint,
    setup,
    warmup_cosine,
)
from h2r.training.world_model import build_model


def load_world_model(run_dir, device):
    """Frozen world model of a training run (config.yaml + best.pt)"""
    run_dir = Path(run_dir)
    if not (run_dir / "best.pt").exists():
        raise FileNotFoundError(f"world_model: no best.pt in {run_dir}")
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(
        torch.load(run_dir / "best.pt", map_location=device)["model"], strict=False
    )
    model.eval().requires_grad_(False)
    return model, cfg


def world_model_info(run_dir):
    """Run directory and git commit of the world model the controller belongs to"""
    meta = Path(run_dir) / "metadata.json"
    return {
        "run": str(run_dir),
        "git": json.load(open(meta)).get("git") if meta.exists() else None,
    }


def read_episodes(root, split):
    tables = [
        pq.read_table(p, columns=["video", "frames", "actions"])
        for p in sorted(Path(root, "frames").glob(f"{split}-*.parquet"))
    ]
    return [row for t in tables for row in t.to_pylist()]


@torch.no_grad()
def encode_episodes(model, episodes, device, chunk=64):
    """Transitions of every episode: s_t (M, N, D) in fp16, a_t (M, act_dim), mu_t (M, d_a)"""
    states, actions, targets = [], [], []
    for ep in episodes:
        if len(ep["actions"]) != len(ep["frames"]) - 1:
            raise ValueError(
                f"{ep['video']}: {len(ep['frames'])} frames but {len(ep['actions'])} actions"
            )
        frames = np.stack([np.asarray(Image.open(io.BytesIO(j))) for j in ep["frames"]])
        px = torch.from_numpy(frames).permute(0, 3, 1, 2)
        emb = torch.cat(
            [
                model.encode({"pixels": px[i : i + chunk].to(device)[None]})["emb"][0]
                for i in range(0, len(px), chunk)
            ]
        )  # (T, N, D)
        mu = model.idm(emb[None])[0][0]  # eval: z = mu, (T - 1, d_a)
        states.append(emb[:-1].half().cpu())
        targets.append(mu.float().cpu())
        actions.append(torch.tensor(ep["actions"], dtype=torch.float32))
    return {"s": torch.cat(states), "a": torch.cat(actions), "mu": torch.cat(targets)}


def cached_transitions(model, cfg, split, device, log):
    """Encoded transitions of a split, computed once per world model run"""
    wm = Path(cfg["world_model"])
    path = (
        Path(cfg["data"]["cache"])
        / f"{wm.parent.name}-{wm.name}"
        / f"{Path(cfg['data']['root']).name}-{split}.pt"
    )
    if path.exists():
        return torch.load(path)
    log(f"encoding {split} with {wm}")
    data = encode_episodes(model, read_episodes(cfg["data"]["root"], split), device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data


def build_controller(cfg, wm_cfg):
    """The controller's dimensions come from the world model, the rest from its own config"""
    idm = wm_cfg["model"]["idm"]
    return Controller(
        num_patches=idm["num_patches"],
        dim=idm["dim"],
        latent_dim=idm.get("action_dim"),
        **cfg,
    )


def controller_loss(ctrl, s, a, mu):
    """L2 to the IDM's latent action, summed over its dimensions"""
    return (ctrl(a, s.float()).float() - mu).pow(2).sum(-1).mean()


@torch.no_grad()
def evaluate(ctrl, data, device, autocast, chunk=4096):
    ctrl.eval()
    total = 0.0
    for i in range(0, len(data["a"]), chunk):
        batch = [data[k][i : i + chunk].to(device) for k in ("s", "a", "mu")]
        with autocast:
            total += controller_loss(ctrl, *batch).item() * len(batch[1])
    ctrl.train()
    return total / len(data["a"])


def train_controller(cfg):
    cfg, run_dir, resume = prepare_run(cfg)
    log = make_logger(run_dir)
    device, autocast = setup(cfg)

    download_data(cfg["data"])
    wm, wm_cfg = load_world_model(cfg["world_model"], device)
    train, val = (
        cached_transitions(wm, cfg, split, device, log) for split in ("train", "val")
    )
    del wm

    ctrl = build_controller(cfg["model"], wm_cfg).to(device)
    ctrl.set_action_stats(train["a"])
    opt = torch.optim.AdamW(
        ctrl.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine(cfg["optim"]["warmup"], cfg["iterations"])
    )

    meta_path = run_dir / "metadata.json"
    step, best = 0, math.inf
    if resume:
        _, step, best = load_checkpoint(
            run_dir / "last.pt", ctrl, [opt], [sched], device
        )
        meta = resumed_metadata(meta_path)
    else:
        sizes = {"train_transitions": len(train["a"]), "val_transitions": len(val["a"])}
        meta = new_metadata(
            cfg, device, sizes, sum(p.numel() for p in ctrl.parameters())
        )
        meta["world_model"] = world_model_info(cfg["world_model"])
    json.dump(meta, open(meta_path, "w"), indent=2)
    wandb = init_wandb(cfg, run_dir, meta, meta_path)

    log(
        f"run {run_dir} | {device} | {meta['trainable_params']:,} trainable params | "
        f"{len(train['a'])} train transitions | {cfg['iterations']} steps"
    )
    gen = torch.Generator().manual_seed(cfg["seed"] + step)
    ctrl.train()
    while step < cfg["iterations"]:
        idx = torch.randint(len(train["a"]), (cfg["batch_size"],), generator=gen)
        with autocast:
            loss = controller_loss(
                ctrl, *(train[k][idx].to(device) for k in ("s", "a", "mu"))
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        if wandb:
            wandb.log(
                {"train/loss": loss.item(), "lr": sched.get_last_lr()[0]}, step=step
            )

        if step % cfg["eval_every"] == 0 or step == cfg["iterations"]:
            val_loss = evaluate(ctrl, val, device, autocast)
            if wandb:
                wandb.log({"val/loss": val_loss}, step=step)
            log(
                f"step {step}/{cfg['iterations']} | train loss {loss.item():.4f} | val loss {val_loss:.4f}"
            )
            if val_loss < best:
                best = val_loss
                save_checkpoint(
                    run_dir / "best.pt", ctrl, [opt], [sched], 0, step, best
                )
            save_checkpoint(run_dir / "last.pt", ctrl, [opt], [sched], 0, step, best)

    log(f"done | best val loss {best:.4f}")
    if wandb:
        wandb.finish()
