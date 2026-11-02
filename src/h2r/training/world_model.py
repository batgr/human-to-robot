import itertools

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from h2r.data.dataset import ClipDataset, shards
from h2r.metrics.latent_actions import LatentActions
from h2r.metrics.prediction import PredictionQuality, PredictorInternals, RolloutError
from h2r.metrics.representation import RepresentationHealth
from h2r.training.utils import (
    grad_norms,
    init_wandb,
    pick_device,
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
    update_ratios,
)
import json
import time
from pathlib import Path
import yaml
from h2r.models.jepa import build_model
from h2r.models.module import SIGReg


def build_loaders(cfg, device):
    clip_len = cfg["data"]["clip_len"]
    root = cfg["data"]["processed_dir"]
    ds_train = ClipDataset(root, "train", clip_len)
    args = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=cfg["num_workers"] > 0,
    )
    train = DataLoader(ds_train, shuffle=True, drop_last=True, **args)
    if cfg["overfit_batches"]:  # the same first batches for training and validation
        batches = list(itertools.islice(train, cfg["overfit_batches"]))
        return batches, batches
    val = None
    if shards(root, "val"):
        ds_val = ClipDataset(root, "val", clip_len)
        order = torch.randperm(
            len(ds_val), generator=torch.Generator().manual_seed(0)
        ).tolist()
        val = DataLoader(ds_val, sampler=order, drop_last=False, **args)
    return train, val


def build_optimizer(model, cfg):
    """AdamW, no weight decay on biases, norms and embeddings"""
    optim = cfg["optim"]
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen encoder
        if p.ndim < 2 or name.endswith(
            ("pos_embedding", "query", "cls_token", "position_embeddings")
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": optim["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=optim["lr"],
    )
    schedule = warmup_cosine(
        int(optim["warmup"] * cfg["iterations"]), cfg["iterations"]
    )
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, schedule)


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
    device = pick_device()

    torch.set_float32_matmul_precision("high")
    autocast = make_autocast(cfg["precision"], device)

    train_loader, val_loader = build_loaders(cfg, device)
    train_iter = forever(train_loader)
    train_probe = list(itertools.islice(train_loader, 2))  # fixed training clips, for the overfitting gap

    model = build_model(cfg["model"]).to(device)
    sigreg = SIGReg().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt, sched = build_optimizer(model, cfg)

    start, best = 1, float("inf")  # best val/loss
    if cfg["resume"]:
        ckpt = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
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

    wandb_run = init_wandb(cfg, run_dir)

    model.train()
    sums, data_time, t0 = {}, 0.0, time.time()

    for step in range(start, cfg["iterations"] + 1):

        t = time.time()
        clip = next(train_iter).to(device, non_blocking=True)
        data_time += time.time() - t

        logging = step % cfg["log_every"] == 0
        if logging:  # the weight matrices before the step, for the update ratios
            before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad and p.ndim == 2}
        out = train_step(model, sigreg, clip, opt, sched, params, cfg, autocast)
        for k, v in out.items():
            sums[k] = sums.get(k, 0) + v.detach()

        if logging:
            n = cfg["log_every"]
            m = {k: v.item() / n for k, v in sums.items()}
            elapsed = time.time() - t0
            metrics = {
                "loss/total": m["loss"],
                "loss/tf": m["tf"],
                "loss/kl": m["kl"],
                "loss/sigreg": m["sigreg"],
                "loss/kl_weighted": cfg["loss"]["beta"]
                * m["kl"],  # weighted terms: their real share of the loss
                "loss/sigreg_weighted": cfg["loss"]["sigreg"] * m["sigreg"],
                **{f"loss/{k}": v for k, v in m.items() if k.startswith("rollout")},
                "optim/lr": sched.get_last_lr()[0],
                **{f"optim/{k}": v for k, v in m.items() if k.startswith("gnorm")},
                "optim/clipped": m["clipped"],  # share of the steps whose gradient was clipped
                **{f"optim/{k}": v for k, v in update_ratios(model, before).items()},
                "system/clips_per_s": n * cfg["batch_size"] / elapsed,
                "system/step_s": elapsed / n,
                "system/data_s": data_time / n,
                "system/mem_gb": peak_memory_gb(device),
            }
            log(run_dir, step, metrics, wandb_run)
            sums, data_time, t0 = {}, 0.0, time.time()

        if step % cfg["eval_every"] == 0 or step == cfg["iterations"]:
            t_eval = time.time()
            if val_loader is not None:
                metrics = evaluate(model, sigreg, val_loader, cfg, device, autocast)
                metrics["prediction/tf_over_persistence_train"] = train_fit(model, train_probe, device, autocast)
                log(run_dir, step, metrics, wandb_run)
                if metrics["loss/total_val"] < best:
                    best = metrics["loss/total_val"]
                    save_checkpoint(
                        run_dir / "best.pt",
                        model=trainable_state(model),
                        step=step,
                        val_loss=best,
                    )
            save_checkpoint(
                run_dir / "last.pt",
                model=trainable_state(model),
                opt=opt.state_dict(),
                sched=sched.state_dict(),
                step=step,
                best=best,
                rng=rng_state(),
            )
            t0 += time.time() - t_eval

    if wandb_run is not None:
        wandb_run.finish()


@torch.no_grad()
def evaluate(model, sigreg, loader, cfg, device, autocast):
    """Validation metrics"""
    model.eval()
    representation, prediction, internals, rollout, latent = (
        RepresentationHealth(),
        PredictionQuality(),
        PredictorInternals(),
        RolloutError(),
        LatentActions(),
    )
    loss, n = 0.0, 0
    for clip in loader:
        clip = clip.to(device, non_blocking=True)
        with autocast:
            emb = model.encode(clip)
            out = losses(model, sigreg, emb, cfg)
            z, _ = model.idm(emb)  # eval: z = mu
            pred = model.predict(emb[:, :-1], z)
            representation.update(emb, out["sigreg"])
            prediction.update(emb, pred, clip)
            internals.update(model, emb, z)
            rollout.update(model, emb, z)
            latent.update(model, emb, z, pred, out["kl"])
        loss += out["loss"].item() * len(clip)
        n += len(clip)
    model.train()
    groups = {
        "representation": representation,
        "prediction": prediction,
        "predictor": internals,
        "rollout": rollout,
        "latent_action": latent,
    }
    metrics = {"loss/total_val": loss / n}
    for name, group in groups.items():
        metrics |= {f"{name}/{k}": v for k, v in group.compute().items()}
    return metrics


@torch.no_grad()
def train_fit(model, batches, device, autocast):
    """tf / persistence on fixed training clips, in eval mode as the validation: the gap between the two is the
    overfitting"""
    model.eval()
    err = pers = 0.0
    for clip in batches:
        with autocast:
            emb = model.encode(clip.to(device, non_blocking=True))
            pred = model.predict(emb[:, :-1], model.idm(emb)[0])
        target = emb[:, 1:].float()
        err += (pred.float() - target).pow(2).mean()
        pers += (emb[:, :-1].float() - target).pow(2).mean()
    model.train()
    return float(err / pers)


def train_step(model, sigreg, clip, opt, sched, params, cfg, autocast):

    with autocast:
        out = losses(model, sigreg, model.encode(clip), cfg)
    opt.zero_grad(set_to_none=True)
    out["loss"].backward()
    out |= grad_norms(model)
    out["gnorm"] = torch.nn.utils.clip_grad_norm_(
        params, cfg["optim"]["grad_clip"]
    )  # norm before clipping
    out["clipped"] = (out["gnorm"] > cfg["optim"]["grad_clip"]).float()
    opt.step()
    sched.step()
    return out


def losses(model, sigreg, emb, cfg):
    beta = cfg["loss"]["beta"]
    sig_w = cfg["loss"]["sigreg"]
    z, kl = model.idm(emb)  # (B, T-1, z_dim), one latent action per transition

    pred = model.predict(emb[:, :-1], z)  # (B, T-1, D) or (B, T-1, N, D)

    tf = F.mse_loss(pred.float(), emb[:, 1:].float())  # every transition of the clip
    rollout = rollout_terms(model, emb, z, cfg["loss"]["rollout_steps"], cfg["loss"]["rollout_window"])

    sig = sigreg(
        rearrange(emb.float(), "b t ... d -> (t ...) b d")
    )  # per frame (and token) across clips

    loss = tf + sum(rollout) + (beta * kl) + (sig_w * sig)

    return {"loss": loss, "tf": tf, "kl": kl, "sigreg": sig} | {
        f"rollout_{k}": term for k, term in enumerate(rollout, start=2)
    }


def rollout_terms(model, emb, z, steps, w):
    """The k-step loss of JEPA-WM (Terver et al. 2025): L_k = error on a state predicted after k predictor steps from
    real states, for k = 2..steps (L_1 is tf). Each step reads the previous predictions detached (last-gradient only)
    in a window of w states (W_t), as at planning time; every state of the clip is a target once per k, so every
    initial context length is used (the paper samples one at random).
    emb: (B, T, D) or (B, T, N, D), z: (B, T-1, z_dim) the latent actions of the true transitions
    """
    T = emb.size(1)
    assert w < T, f"rollout_window {w} needs clips of more than {w} frames"
    depth = [
        emb
    ]  # depth[d][:, p]: state p after d steps from the real state p - d, detached (d = 0: real)
    terms = []
    for k in range(1, steps + 1 if steps > 1 else 1):
        ctx, lat, read = [], [], []
        for j in range(k, T):  # target j, start j - k
            a = max(
                0, j - w
            )  # window a..a+w-1; when j < w the causal mask hides the states after j - 1
            ctx.append(
                torch.stack(
                    [
                        depth[p - j + k][:, p] if j - k < p < j else emb[:, p]
                        for p in range(a, a + w)
                    ],
                    dim=1,
                )
            )
            lat.append(z[:, a : a + w])
            read.append(j - 1 - a)  # output that predicts state j
        out = model.predict(torch.cat(ctx), torch.cat(lat)).unflatten(
            0, (len(read), -1)
        )
        pred = torch.stack(
            [out[i, :, r] for i, r in enumerate(read)], dim=1
        )  # (B, T-k, ...)
        if k > 1:
            terms.append(F.mse_loss(pred.float(), emb[:, k:].float()))
        depth.append(
            torch.cat([emb[:, :k], pred], dim=1).detach()
        )  # positions < k never read at depth k
    return terms
