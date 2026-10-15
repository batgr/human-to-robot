"""Train the JEPA world model and the IDM on video clips."""

import json
import math
import random

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader

from h2r.data.dataset import ClipDataset
from h2r.models.encoder import FrozenEncoder
from h2r.models.idm import IDM
from h2r.models.jepa import JEPA
from h2r.models.module import MLP, ARPredictor, PatchProjector, SIGReg
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


def build_model(cfg):
    return JEPA(
        FrozenEncoder(**cfg["encoder"]),
        ARPredictor(**cfg["predictor"]),
        IDM(**cfg["idm"]),
        PatchProjector(**cfg["projector"]),
        MLP(**cfg["pred_proj"]),
    )


def build_optimizers(model, cfg, total_steps):
    """Muon for the 2D weights inside Transformer blocks, AdamW for everything else"""
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen encoder
        inside_block = ".layers." in name or ".readout." in name
        (muon if p.ndim == 2 and inside_block else adamw).append(p)
    opts = [
        torch.optim.Muon(muon, lr=cfg["muon_lr"], weight_decay=cfg["weight_decay"]),
        torch.optim.AdamW(adamw, lr=cfg["adamw_lr"], weight_decay=cfg["weight_decay"]),
    ]
    schedule = warmup_cosine(int(cfg["warmup"] * total_steps), total_steps)
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, schedule) for o in opts]
    return opts, scheds


def losses(model, sigreg, clip, cfg, t=None):
    """clip: (B, C + K, 3, H, W) uint8 -> dict of losses"""
    C, K = cfg["context"], cfg["rollout_steps"]
    emb = model.encode({"pixels": clip})["emb"]  # (B, C + K, N, D)
    z, kl = model.idm(emb)  # (B, C + K - 1, D)

    # teacher forcing: frames 1..C predict frames 2..C+1
    pred = model.predict(emb[:, :C], z[:, :C])
    tf = F.mse_loss(pred.float(), emb[:, 1 : C + 1].float())

    # rollout (JEPA-WM): random real prefix of t frames, then predict from own predictions.
    # The first step equals a teacher forcing output (frame-causal mask), so it is reused.
    t = t or random.randint(1, C)
    ctx = torch.cat([emb[:, :t], pred[:, t - 1 : t]], dim=1)
    ro = []
    for k in range(1, K):
        idx = t + k  # frame to predict
        window = ctx[
            :, -C:
        ].detach()  # TBPTT: no gradient through the inputs, z keeps its gradient
        p = model.predict(window, z[:, idx - window.size(1) : idx])[:, -1:]
        ro.append(F.mse_loss(p.float(), emb[:, idx : idx + 1].float()))
        ctx = torch.cat([ctx, p], dim=1)
    ro = torch.stack(ro).mean()

    sig = sigreg(rearrange(emb.float(), "b t n d -> (t n) b d"))
    loss = tf + ro + cfg["loss"]["beta"] * kl + cfg["loss"]["sigreg"] * sig
    return {"loss": loss, "tf": tf, "ro": ro, "kl": kl, "sigreg": sig}, emb, z


@torch.no_grad()
def evaluate(model, sigreg, loader, cfg, device, autocast):
    """Validation losses with a full real context (as in planning)"""
    model.eval()
    C, sums, n = cfg["context"], {}, 0
    for clip in loader:
        clip = clip.to(device, non_blocking=True)
        with autocast:
            out, _, _ = losses(model, sigreg, clip, cfg, t=C)
        for k, v in out.items():
            sums[k] = sums.get(k, 0) + v.item() * len(clip)
        n += len(clip)
    model.train()
    return {k: v / n for k, v in sums.items()}


def build_loaders(data, batch_size, device):
    """Train and validation clips"""
    download_data(data)
    args = dict(
        batch_size=batch_size,
        num_workers=data["num_workers"],
        pin_memory=device == "cuda",
        persistent_workers=data["num_workers"] > 0,
    )
    train_loader = DataLoader(
        ClipDataset(data["root"], "train", data["clip_len"]),
        shuffle=True,
        drop_last=True,
        **args,
    )
    val_loader = DataLoader(ClipDataset(data["root"], "val", data["clip_len"]), **args)
    return train_loader, val_loader


def train_step(model, sigreg, clip, cfg, opts, scheds, trainable, autocast):
    with autocast:
        out, _, _ = losses(model, sigreg, clip, cfg)
    for o in opts:
        o.zero_grad(set_to_none=True)
    out["loss"].backward()
    torch.nn.utils.clip_grad_norm_(trainable, cfg["optim"]["grad_clip"])
    for o, s in zip(opts, scheds):
        o.step()
        s.step()
    return out


def train(cfg):
    cfg, run_dir, resume = prepare_run(cfg)
    log = make_logger(run_dir)
    device, autocast = setup(cfg)
    train_loader, val_loader = build_loaders(cfg["data"], cfg["batch_size"], device)

    model = build_model(cfg["model"]).to(device)
    sigreg = SIGReg().to(device)
    total_steps = cfg["epochs"] * len(train_loader)
    opts, scheds = build_optimizers(model, cfg["optim"], total_steps)
    trainable = [p for p in model.parameters() if p.requires_grad]

    meta_path = run_dir / "metadata.json"
    epoch, step, best = 0, 0, math.inf
    if resume:
        epoch, step, best = load_checkpoint(
            run_dir / "last.pt", model, opts, scheds, device
        )
        meta = resumed_metadata(meta_path)
    else:
        sizes = {
            "train_clips": len(train_loader.dataset),
            "val_clips": len(val_loader.dataset),
        }
        meta = new_metadata(cfg, device, sizes, sum(p.numel() for p in trainable))
    json.dump(meta, open(meta_path, "w"), indent=2)
    wandb = init_wandb(cfg, run_dir, meta, meta_path)

    log(
        f"run {run_dir} | {device} | {meta['trainable_params']:,} trainable params | "
        f"{len(train_loader.dataset)} train clips | {total_steps} steps"
    )
    model.train()
    while epoch < cfg["epochs"]:
        for clip in train_loader:
            out = train_step(
                model,
                sigreg,
                clip.to(device, non_blocking=True),
                cfg,
                opts,
                scheds,
                trainable,
                autocast,
            )
            step += 1
            if wandb:
                wandb.log(
                    {f"train/{k}": v.item() for k, v in out.items()}
                    | {
                        "lr/muon": scheds[0].get_last_lr()[0],
                        "lr/adamw": scheds[1].get_last_lr()[0],
                    },
                    step=step,
                )

        epoch += 1
        val = evaluate(model, sigreg, val_loader, cfg, device, autocast)
        if wandb:
            wandb.log({f"val/{k}": v for k, v in val.items()}, step=step)
        log(
            f"epoch {epoch}/{cfg['epochs']} step {step} | "
            + " | ".join(f"{k} {v:.4f}" for k, v in val.items())
        )
        if val["ro"] < best:
            best = val["ro"]
            save_checkpoint(run_dir / "best.pt", model, opts, scheds, epoch, step, best)
        save_checkpoint(run_dir / "last.pt", model, opts, scheds, epoch, step, best)

    log(f"done | best val rollout {best:.4f}")
    if wandb:
        wandb.finish()
