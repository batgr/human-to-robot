"""Train the JEPA world model and the IDM on video clips."""

import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader

from h2r.data.dataset import ClipDataset
from h2r.models.jepa import build_model
from h2r.models.module import SIGReg
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


def rollout_steps(cfg, step):
    """K at a training step (curriculum): from start to end, one more step at regular intervals over the first
    ramp fraction of the iterations, then end. Accuracy on one step first, robustness to own predictions after"""
    k = cfg["rollout_steps"]
    ramp = k["ramp"] * cfg["iterations"]
    if step >= ramp:
        return k["end"]
    return k["start"] + int(step / ramp * (k["end"] - k["start"] + 1))


def losses(model, sigreg, emb, cfg, K, t=None):
    """emb: (B, C + K, N, D) encoded clip, K rollout steps (1: teacher forcing only) -> dict of losses"""
    C = cfg["context"]
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
    ro = torch.stack(ro).mean() if ro else tf.new_zeros(())

    sig = sigreg(rearrange(emb.float(), "b t n d -> (t n) b d"))
    loss = tf + ro + cfg["loss"]["beta"] * kl + cfg["loss"]["sigreg"] * sig
    return {"loss": loss, "tf": tf, "ro": ro, "kl": kl, "sigreg": sig}


def effective_rank(x):
    """RankMe (Garrido et al. 2023): exp of the entropy of the normalized singular values of x (M, D)"""
    s = torch.linalg.svdvals(x.float().cpu())
    p = s / s.sum() + 1e-7
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def evaluate(model, sigreg, loader, cfg, device, autocast):
    """Validation with a full real context (as in planning), on clips of context + val_horizon frames:
    - the losses, on the first context + K frames, K the last rollout_steps of the curriculum;
    - skill(h) = 1 - err(h) / err of persistence (the last context state repeated), the predictor unrolled h steps
      with the IDM's latent actions (oracle). Sums over the whole split, so near-static clips do not blow it up;
    - controllability = 1 - err(z) / err(z of another clip), teacher forcing: 0 when z is ignored;
    - effective rank of the states (collapse) and active latent dims (Var(mu) > 0.01, Burda et al. 2016)."""
    model.eval()
    C, K, H = cfg["context"], cfg["rollout_steps"]["end"], cfg["val_horizon"]
    sums, n, ranks = {}, 0, []
    err, persist, shuffled = torch.zeros(H), torch.zeros(H), torch.zeros(2)
    mu_sum = mu_sq = n_mu = 0
    for clip in loader:
        clip = clip.to(device, non_blocking=True)
        with autocast:
            emb = model.encode(clip)  # (B, C + H, N, D)
            out = losses(model, sigreg, emb[:, : C + K], cfg, K, t=C)
            z, _ = model.idm(emb)  # eval: z = mu
            pred = model.rollout(emb[:, :C], z[:, : C - 1], z[:, C - 1 :][:, None], history_size=C)[:, 0]  # (B, H, N, D)
            tf = model.predict(emb[:, :C], z[:, :C])
            tf_other = model.predict(emb[:, :C], z[:, :C].roll(1, dims=0))
        emb, z = emb.float(), z.float()
        err += (pred.float() - emb[:, C:]).pow(2).sum(dim=(0, 2, 3)).cpu()
        persist += (emb[:, C - 1 : C] - emb[:, C:]).pow(2).sum(dim=(0, 2, 3)).cpu()
        target = emb[:, 1 : C + 1]
        shuffled += torch.stack([(tf.float() - target).pow(2).sum(), (tf_other.float() - target).pow(2).sum()]).cpu()
        ranks.append(effective_rank(emb.flatten(0, 2)))
        mu = z.flatten(0, 1)
        mu_sum, mu_sq, n_mu = mu_sum + mu.sum(0), mu_sq + mu.pow(2).sum(0), n_mu + len(mu)
        for k, v in out.items():
            sums[k] = sums.get(k, 0) + v.item() * len(clip)
        n += len(clip)
    model.train()
    metrics = {k: v / n for k, v in sums.items()}
    skill = 1 - err / persist
    metrics |= {f"skill_h{h + 1}": float(skill[h]) for h in range(H)} | {"skill": float(skill.mean())}
    metrics["controllability"] = float(1 - shuffled[0] / shuffled[1])
    metrics["rank"] = sum(ranks) / len(ranks)
    var = mu_sq / n_mu - (mu_sum / n_mu).pow(2)
    metrics["active_dims"] = int((var > 0.01).sum())
    return metrics


def build_loaders(cfg, device):
    """Train clips of context + K frames (K the last rollout_steps), validation clips of context + val_horizon frames"""
    data = cfg["data"]
    download_data(data)
    args = dict(
        batch_size=cfg["batch_size"],
        num_workers=data["num_workers"],
        pin_memory=device == "cuda",
        persistent_workers=data["num_workers"] > 0,
    )
    subset = dict(include=data.get("include"), max_videos=data.get("max_videos"), seed=data["seed"])
    train_loader = DataLoader(
        ClipDataset(data["root"], "train", cfg["context"] + cfg["rollout_steps"]["end"], **subset),
        shuffle=True,
        drop_last=True,
        **args,
    )
    val_len = cfg["context"] + cfg["val_horizon"]
    val_loader = DataLoader(ClipDataset(data["root"], "val", val_len, include=data.get("include")), **args)
    return train_loader, val_loader


def train_step(model, sigreg, clip, cfg, K, opts, scheds, trainable, autocast):
    with autocast:
        out = losses(model, sigreg, model.encode(clip[:, : cfg["context"] + K]), cfg, K)
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
    if cfg["val_horizon"] < cfg["rollout_steps"]["end"]:
        raise ValueError("val_horizon must cover the last rollout_steps (validation losses at K = end)")
    train_loader, val_loader = build_loaders(cfg, device)

    model = build_model(cfg["model"]).to(device)
    if cfg.get("init_from") and not resume:  # weights of a trained world model run, e.g. SSv2 before robot videos
        state = torch.load(Path(cfg["init_from"]) / "best.pt", map_location=device)["model"]
        model.load_state_dict(state, strict=False)  # the frozen encoder is not in checkpoints
    sigreg = SIGReg().to(device)
    opts, scheds = build_optimizers(model, cfg["optim"], cfg["iterations"])
    trainable = [p for p in model.parameters() if p.requires_grad]

    meta_path = run_dir / "metadata.json"
    step, best = 0, -math.inf
    if resume:
        _, step, best = load_checkpoint(run_dir / "last.pt", model, opts, scheds, device)
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
        f"{len(train_loader.dataset)} train clips ({len(train_loader)} steps per epoch) | {cfg['iterations']} steps"
    )
    model.train()
    while step < cfg["iterations"]:
        for clip in train_loader:
            K = rollout_steps(cfg, step)
            out = train_step(
                model,
                sigreg,
                clip.to(device, non_blocking=True),
                cfg,
                K,
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
                        "train/K": K,
                        "lr/muon": scheds[0].get_last_lr()[0],
                        "lr/adamw": scheds[1].get_last_lr()[0],
                    },
                    step=step,
                )

            if step % cfg["eval_every"] == 0 or step == cfg["iterations"]:
                val = evaluate(model, sigreg, val_loader, cfg, device, autocast)
                if wandb:
                    wandb.log({f"val/{k}": v for k, v in val.items()}, step=step)
                log(f"step {step}/{cfg['iterations']} K {K} | " + " | ".join(f"{k} {v:.4f}" for k, v in val.items()))
                if val["skill"] > best:  # mean skill over the horizons: a collapsing space also lowers the raw errors
                    best = val["skill"]
                    save_checkpoint(run_dir / "best.pt", model, opts, scheds, 0, step, best)
                save_checkpoint(run_dir / "last.pt", model, opts, scheds, 0, step, best)
            if step == cfg["iterations"]:
                break

    log(f"done | best val skill {best:.4f}")
    if wandb:
        wandb.finish()
