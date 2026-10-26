import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from h2r.data.dataset import ClipDataset, shards
from h2r.training.utils import (
    seed_everything,
    make_autocast,
    make_run_dir,
    forever,
    warmup_cosine,
    peak_memory_gb,
    log,
    rng_state,
    set_rng_state,
    save_checkpoint,
)
import json
import time
from pathlib import Path
import yaml
from h2r.models.jepa import build_model
from h2r.models.module import SIGReg


def build_loaders(cfg, device):
    clip_len = cfg["context"] + cfg["rollout_steps"]
    root = cfg["data"]["processed_dir"]
    ds_train = ClipDataset(root, "train", clip_len)
    args = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=cfg["num_workers"] > 0,
    )
    train = DataLoader(ds_train, shuffle=True, drop_last=True, **args)
    val = None
    if shards(root, "val"):  # no validation split in the local data
        ds_val = ClipDataset(root, "val", clip_len)
        val = DataLoader(ds_val, shuffle=False, drop_last=False, **args)
    return train, val


def build_optimizers(model, cfg):
    """Muon for the 2D weights inside Transformer blocks, AdamW for the rest"""
    optim = cfg["optim"]
    muon, decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen encoder
        if p.ndim == 2 and (".layers." in name or ".readout." in name):
            muon.append(p)
        elif p.ndim < 2 or name.endswith(
            ("pos_embedding", "query", "cls_token", "position_embeddings")
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    opts = [
        torch.optim.Muon(muon, lr=optim["muon_lr"], weight_decay=optim["weight_decay"]),
        torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": optim["weight_decay"]},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=optim["adamw_lr"],
        ),
    ]
    schedule = warmup_cosine(
        int(optim["warmup"] * cfg["iterations"]), cfg["iterations"]
    )
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, schedule) for o in opts]
    return opts, scheds


def trainable_state(model):
    """State dict without the frozen weights (a frozen encoder reloads from the Hub), buffers kept"""
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k not in frozen}


def train(cfg):
    if cfg["resume"]:
        run_dir = Path(cfg["resume"])
        cfg = yaml.safe_load(open(run_dir / "config.yaml"))  # the run's own config
        cfg["resume"] = str(run_dir)
    seed_everything(cfg["seed"])
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )

    torch.set_float32_matmul_precision("high")
    autocast = make_autocast(cfg["precision"], device)

    train_loader, val_loader = build_loaders(cfg, device)
    train_iter = forever(train_loader)

    model = build_model(cfg["model"]).to(device)
    sigreg = SIGReg().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opts, scheds = build_optimizers(model, cfg)

    start, best = 1, float("inf")  # best val/loss
    if cfg["resume"]:
        ckpt = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        for o, state in zip(opts, ckpt["opts"]):
            o.load_state_dict(state)
        for s, state in zip(scheds, ckpt["scheds"]):
            s.load_state_dict(state)
        set_rng_state(ckpt["rng"])
        start, best = ckpt["step"] + 1, ckpt["best"]
        metrics_file = (
            run_dir / "metrics.jsonl"
        )  # drop what was logged after the checkpoint
        lines = metrics_file.read_text().splitlines(keepends=True)
        metrics_file.write_text(
            "".join(line for line in lines if json.loads(line)["step"] < start)
        )
    else:
        run_dir = make_run_dir(cfg)

    print(
        f"run: {run_dir} | device: {device} | trainable params: {sum(p.numel() for p in params)}"
    )

    wandb_run = None
    if cfg["wandb"]["enabled"]:
        import wandb

        wandb_run = wandb.init(
            project=cfg["wandb"]["project"],
            name=run_dir.name,
            id=run_dir.name,  # unique (timestamp): a resumed run continues the same W&B run
            resume="allow",
            config=cfg,
            dir=run_dir,
        )

    model.train()
    sums, data_time, t0 = {}, 0.0, time.time()  # accumulated over log_every steps

    for step in range(start, cfg["iterations"] + 1):

        t = time.time()
        clip = next(train_iter).to(device, non_blocking=True)
        data_time += time.time() - t

        out = train_step(model, sigreg, clip, opts, scheds, params, cfg, autocast)
        for k, v in out.items():
            sums[k] = sums.get(k, 0) + v.detach()

        if step % cfg["log_every"] == 0:
            n = cfg["log_every"]
            metrics = {k: v.item() / n for k, v in sums.items()}
            elapsed = time.time() - t0
            metrics |= {
                "beta_kl": cfg["loss"]["beta"]
                * metrics["kl"],  # weighted terms: their real share of the loss
                "sigreg_w": cfg["loss"]["sigreg"] * metrics["sigreg"],
                "lr_muon": scheds[0].get_last_lr()[0],
                "lr_adamw": scheds[1].get_last_lr()[0],
                "clips_per_s": n * cfg["batch_size"] / elapsed,
                "step_s": elapsed / n,
                "data_s": data_time / n,
                "mem_gb": peak_memory_gb(device),
            }
            log(run_dir, step, metrics, wandb_run)
            sums, data_time, t0 = {}, 0.0, time.time()

        if step % cfg["eval_every"] == 0 or step == cfg["iterations"]:
            t_eval = time.time()
            if val_loader is not None:
                metrics = evaluate(model, sigreg, val_loader, cfg, device, autocast)
                log(run_dir, step, metrics, wandb_run)
                if metrics["val/loss"] < best:
                    best = metrics["val/loss"]
                    save_checkpoint(
                        run_dir / "best.pt",
                        model=trainable_state(model),
                        step=step,
                        val_loss=best,
                    )
            save_checkpoint(
                run_dir / "last.pt",
                model=trainable_state(model),
                opts=[o.state_dict() for o in opts],
                scheds=[s.state_dict() for s in scheds],
                step=step,
                best=best,
                rng=rng_state(),
            )
            t0 += (
                time.time() - t_eval
            )  # evaluation and saving are not training throughput

    if wandb_run is not None:
        wandb_run.finish()


@torch.no_grad()
def evaluate(model, sigreg, loader, cfg, device, autocast):

    model.eval()
    sums, n = {}, 0
    for clip in loader:
        clip = clip.to(device, non_blocking=True)
        b = clip.shape[0]
        with autocast:
            emb = model.encode(clip)
            out = losses(model, sigreg, emb, cfg)
        out["straightness"] = straightness(emb)
        for k, v in out.items():
            sums[k] = sums.get(k, 0) + v.float() * b
        n += b
    model.train()
    return {f"val/{k}": v.item() / n for k, v in sums.items()}


def train_step(model, sigreg, clip, opts, scheds, params, cfg, autocast):

    with autocast:
        out = losses(model, sigreg, model.encode(clip), cfg)
    for o in opts:
        o.zero_grad(set_to_none=True)
    out["loss"].backward()
    out["gnorm"] = torch.nn.utils.clip_grad_norm_(
        params, cfg["optim"]["grad_clip"]
    )  # norm before clipping
    for o, s in zip(opts, scheds):
        o.step()
        s.step()
    return out


def losses(model, sigreg, emb, cfg):
    beta = cfg["loss"]["beta"]
    sig_w = cfg["loss"]["sigreg"]
    z, kl = model.idm(emb)  # (B, T-1, D), one latent action per transition

    pred = model.predict(emb[:, :-1], z)  # (B, T-1, N, D)

    tf = F.mse_loss(pred.float(), emb[:, 1:].float())  # every transition of the clip

    sig = sigreg(rearrange(emb.float(), "b t n d -> (t n) b d"))

    loss = tf + (beta * kl) + (sig_w * sig)

    return {"loss": loss, "tf": tf, "kl": kl, "sigreg": sig}


def straightness(emb):
    """Temporal straightness (LeWM eq. 9): mean cosine similarity between consecutive
    latent velocities v_t = z_t+1 - z_t, a frame being all its tokens; 1 = straight path"""
    v = (emb[:, 1:] - emb[:, :-1]).flatten(2).float()  # (B, T-1, N*D)
    return F.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1).mean()
