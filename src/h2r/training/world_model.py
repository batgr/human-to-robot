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
)
import time
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


def train(cfg):
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

    run_dir = make_run_dir(cfg)

    model = build_model(cfg["model"]).to(device)
    sigreg = SIGReg().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opts, scheds = build_optimizers(model, cfg)

    print(
        f"run: {run_dir} | device: {device} | trainable params: {sum(p.numel() for p in params)}"
    )

    wandb_run = None
    if cfg["wandb"]["enabled"]:
        import wandb

        wandb_run = wandb.init(
            project=cfg["wandb"]["project"], name=run_dir.name, config=cfg, dir=run_dir
        )

    model.train()
    sums, data_time, t0 = {}, 0.0, time.time()  # accumulated over log_every steps

    for step in range(1, cfg["iterations"] + 1):

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

        if val_loader is not None and step % cfg["eval_every"] == 0:
            t_eval = time.time()
            log(
                run_dir,
                step,
                evaluate(model, sigreg, val_loader, cfg, device, autocast),
                wandb_run,
            )
            t0 += time.time() - t_eval  # the evaluation time is not training throughput

    if wandb_run is not None:
        wandb_run.finish()


@torch.no_grad()
def evaluate(model, sigreg, loader, cfg, device, autocast):

    model.eval()
    sums, n = {}, 0
    for clip in loader:
        clip = clip.to(device, non_blocking=True)
        with autocast:
            out = losses(model, sigreg, model.encode(clip), cfg)
        for k, v in out.items():
            sums[k] = sums.get(k, 0) + v.float() * len(clip)  # batch means -> sums
        n += len(clip)
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
    C = cfg["context"]
    beta = cfg["loss"]["beta"]
    sig_w = cfg["loss"]["sigreg"]
    z, kl = model.idm(emb)

    pred = model.predict(emb[:, :C], z[:, :C])  # (B, C, N, D)

    tf = F.mse_loss(pred.float(), emb[:, 1 : C + 1].float())

    sig = sigreg(rearrange(emb.float(), "b t n d -> (t n) b d"))

    loss = tf + (beta * kl) + (sig_w * sig)

    return {"loss": loss, "tf": tf, "kl": kl, "sigreg": sig}
