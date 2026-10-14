"""Train the JEPA world model and the IDM on video clips.

Loss: teacher forcing + rollout (JEPA-WM) + beta * KL + lambda * SIGReg.
"""

import json
import math
import random
import socket
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from huggingface_hub import HfApi, snapshot_download
from torch.utils.data import DataLoader

from h2r.data.dataset import ClipDataset
from h2r.models.encoder import FrozenEncoder
from h2r.models.idm import IDM
from h2r.models.jepa import JEPA
from h2r.models.module import MLP, ARPredictor, PatchProjector, SIGReg


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
    warm = int(cfg["warmup"] * total_steps)

    def schedule(step):  # linear warmup, then cosine
        if step < warm:
            return step / max(warm, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total_steps - warm, 1)))

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
        window = ctx[:, -C:].detach()  # TBPTT: no gradient through the inputs, z keeps its gradient
        p = model.predict(window, z[:, idx - window.size(1) : idx])[:, -1:]
        ro.append(F.mse_loss(p.float(), emb[:, idx : idx + 1].float()))
        ctx = torch.cat([ctx, p], dim=1)
    ro = torch.stack(ro).mean()

    sig = sigreg(rearrange(emb.float(), "b t n d -> (t n) b d"))
    loss = tf + ro + cfg["loss"]["beta"] * kl + cfg["loss"]["sigreg"] * sig
    return {"loss": loss, "tf": tf, "ro": ro, "kl": kl, "sigreg": sig}, emb, z


def effective_rank(x):
    """exp(entropy) of the normalized singular values of x: (M, D)"""
    s = torch.linalg.svdvals((x - x.mean(0)).float().cpu())
    p = s / s.sum()
    return torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item()


@torch.no_grad()
def evaluate(model, sigreg, loader, cfg, device, autocast):
    """Validation losses with a full real context (as in planning), baselines and health checks"""
    model.eval()
    C, sums, n = cfg["context"], {}, 0
    for clip in loader:
        clip = clip.to(device, non_blocking=True)
        with autocast:
            out, emb, z = losses(model, sigreg, clip, cfg, t=C)
            target = emb[:, 1 : C + 1].float()
            z_zero = model.predict(emb[:, :C], torch.zeros_like(z[:, :C]))
            z_shuf = model.predict(emb[:, :C], z[torch.randperm(len(z)), :C])
        stats = {k: v.item() for k, v in out.items()} | {
            "persistence": F.mse_loss(emb[:, :C].float(), target).item(),  # nothing moves, 1 step
            "persistence_2": F.mse_loss(emb[:, C - 1].float(), emb[:, C + 1].float()).item(),  # 2 steps
            "z_zero": F.mse_loss(z_zero.float(), target).item(),
            "z_shuffled": F.mse_loss(z_shuf.float(), target).item(),
            "emb_std": emb.float().std(0).mean().item(),
            "emb_rank": effective_rank(rearrange(emb, "b t n d -> (b t n) d")),
            "z_active_dims": (z.float().std(0).mean(0) > 0.1).sum().item(),
        }
        for k, v in stats.items():
            sums[k] = sums.get(k, 0) + v * len(clip)
        n += len(clip)
    model.train()
    return {k: v / n for k, v in sums.items()}


def git_commit():
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def train(cfg):
    resume = cfg.pop("resume", None)
    if resume:  # continue a run with its own config
        run_dir = Path(resume)
        cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    else:
        if isinstance(cfg["model"], str):
            cfg["model"] = yaml.safe_load(open(cfg["model"]))  # the run keeps the full config
        run_dir = Path(cfg["output_dir"]) / cfg["name"] / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True)
        cfg.pop("config_path", None)
        yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"), sort_keys=False)

    log_file = open(run_dir / "log.txt", "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    autocast = torch.autocast(device, dtype=torch.bfloat16, enabled=cfg["precision"] == "bf16")

    # data: download the shards once if they are not there
    data = cfg["data"]
    if not (Path(data["root"]) / "manifest.parquet").exists():
        snapshot_download(data["hf_repo"], repo_type="dataset", local_dir=data["root"])
    loader_args = dict(batch_size=cfg["batch_size"], num_workers=data["num_workers"],
                       pin_memory=device == "cuda", persistent_workers=data["num_workers"] > 0)
    train_loader = DataLoader(ClipDataset(data["root"], "train", data["clip_len"]), shuffle=True,
                              drop_last=True, **loader_args)
    val_loader = DataLoader(ClipDataset(data["root"], "val", data["clip_len"]), **loader_args)

    model = build_model(cfg["model"]).to(device)
    sigreg = SIGReg().to(device)
    total_steps = cfg["epochs"] * len(train_loader)
    opts, scheds = build_optimizers(model, cfg["optim"], total_steps)
    trainable = [p for p in model.parameters() if p.requires_grad]

    meta_path = run_dir / "metadata.json"
    epoch, step, best = 0, 0, math.inf
    if resume:
        ckpt = torch.load(run_dir / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)  # the frozen encoder is not saved
        for o, s, so, ss in zip(opts, scheds, ckpt["opts"], ckpt["scheds"]):
            o.load_state_dict(so)
            s.load_state_dict(ss)
        epoch, step, best = ckpt["epoch"], ckpt["step"], ckpt["best"]
        meta = json.load(open(meta_path))
        meta["resumed"] = meta.get("resumed", []) + [time.strftime("%Y-%m-%d %H:%M:%S")]
    else:
        try:
            data_revision = HfApi().dataset_info(data["hf_repo"]).sha
        except Exception:
            data_revision = None
        meta = {"seed": cfg["seed"], "git": git_commit(), "host": socket.gethostname(), "device": device,
                "torch": torch.__version__, "hf_repo": data["hf_repo"], "data_revision": data_revision,
                "train_clips": len(train_loader.dataset), "val_clips": len(val_loader.dataset),
                "trainable_params": sum(p.numel() for p in trainable)}
    json.dump(meta, open(meta_path, "w"), indent=2)

    wandb = None
    if cfg["wandb"]["enabled"]:
        import wandb

        run = wandb.init(project=cfg["wandb"]["project"], name=f"{cfg['name']}/{run_dir.name}", config=cfg,
                         dir=run_dir, id=meta.get("wandb_id"), resume="allow")
        meta["wandb_id"] = run.id
        json.dump(meta, open(meta_path, "w"), indent=2)

    def save(name):
        state = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
        torch.save({"model": state, "opts": [o.state_dict() for o in opts],
                    "scheds": [s.state_dict() for s in scheds], "epoch": epoch, "step": step, "best": best},
                   run_dir / name)

    log(f"run {run_dir} | {device} | {meta['trainable_params']:,} trainable params | "
        f"{len(train_loader.dataset)} train clips | {total_steps} steps")
    model.train()
    while epoch < cfg["epochs"]:
        t_data = time.perf_counter()
        for clip in train_loader:
            clip = clip.to(device, non_blocking=True)
            t_step = time.perf_counter()
            with autocast:
                out, _, _ = losses(model, sigreg, clip, cfg)
            for o in opts:
                o.zero_grad(set_to_none=True)
            out["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg["optim"]["grad_clip"])
            for o, s in zip(opts, scheds):
                o.step()
                s.step()
            step += 1
            if wandb:
                wandb.log({f"train/{k}": v.item() for k, v in out.items()} | {
                    "train/grad_norm": grad_norm.item(), "lr/muon": scheds[0].get_last_lr()[0],
                    "lr/adamw": scheds[1].get_last_lr()[0], "time/data": t_step - t_data,
                    "time/step": time.perf_counter() - t_step}, step=step)
            t_data = time.perf_counter()

        epoch += 1
        val = evaluate(model, sigreg, val_loader, cfg, device, autocast)
        if wandb:
            wandb.log({f"val/{k}": v for k, v in val.items()}, step=step)
        log(f"epoch {epoch}/{cfg['epochs']} step {step} | tf {val['tf']:.4f} (persistence {val['persistence']:.4f}, "
            f"z=0 {val['z_zero']:.4f}) | ro {val['ro']:.4f} (persistence {val['persistence_2']:.4f}) | "
            f"kl {val['kl']:.2f} | emb std {val['emb_std']:.3f} rank {val['emb_rank']:.1f}")
        if val["ro"] < best:
            best = val["ro"]
            save("best.pt")
        save("last.pt")

    log(f"done | best val rollout {best:.4f}")
    if wandb:
        wandb.finish()
