import itertools
import re

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from h2r.data.dataset import ClipDataset, shards
from h2r.metrics.latent_actions import LatentActions
from h2r.metrics.probe import HandProbe, MotionProbe
from h2r.metrics.prediction import PredictionQuality, PredictorInternals, RolloutError
from h2r.metrics.representation import RepresentationHealth
from h2r.training.utils import (
    build_optimizer,
    epoch_of,
    forever,
    log,
    open_run,
    optimizer_step,
    progress,
    resume,
    save_checkpoint,
    save_last,
    setup,
    trainable_state,
    update_ratios,
)
import time
from pathlib import Path
import yaml
from h2r.models.jepa import build_model
from h2r.models.module import SIGReg


def build_loaders(cfg, device):
    clip_len = cfg["data"]["clip_len"]
    root = cfg["data"]["processed_dir"]
    ds_train = ClipDataset(root, "train", clip_len, cfg["data"].get("stride", 1))  # val: every start, comparable runs
    args = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=cfg["num_workers"] > 0,
    )
    train = DataLoader(ds_train, shuffle=True, drop_last=True, **args)
    val = None
    if shards(root, "val"):
        ds_val = ClipDataset(root, "val", clip_len)
        order = torch.randperm(
            len(ds_val), generator=torch.Generator().manual_seed(0)
        ).tolist()
        val = DataLoader(ds_val, sampler=order, drop_last=False, **args)
    return train, val


def rollout_start(loss, epoch_steps):
    """First step of the k-step loss, teacher forcing only before: the first step of epoch loss.rollout_from_epoch,
    or loss.rollout_from, a step, for runs saved before that key"""
    epoch = loss.get("rollout_from_epoch")
    if epoch is not None and "rollout_from" in loss:
        raise ValueError("loss.rollout_from (a step, runs before rollout_from_epoch) and loss.rollout_from_epoch: give one")
    return epoch * epoch_steps + 1 if epoch is not None else loss.get("rollout_from", 0)


def train(cfg):
    if cfg["resume"]:
        run_dir = Path(cfg["resume"])
        cfg = yaml.safe_load(open(run_dir / "config.yaml"))  # the run's own config
        cfg["resume"] = str(run_dir)
    device, autocast = setup(cfg)
    torch.set_float32_matmul_precision("high")

    train_loader, val_loader = build_loaders(cfg, device)
    train_iter = forever(train_loader)
    train_probe = list(itertools.islice(train_loader, 2))  # fixed training clips, for the overfitting gap
    hands = cfg.get("probe", {}).get("hands")  # tracked hands of the validation videos: HandProbe at each validation
    hand_probe = HandProbe(cfg["data"]["processed_dir"], hands, cfg["probe"]["frames"]) if hands else None
    videos = cfg.get("probe", {}).get("videos")  # whole validation videos: MotionProbe at each validation
    motion_probe = MotionProbe(cfg["data"]["processed_dir"], videos, hands) if videos and val_loader is not None else None

    model = build_model(cfg["model"]).to(device)
    sigreg = SIGReg().to(device)
    opt, sched = build_optimizer(model, cfg["optim"], cfg["iterations"])

    best_metric = cfg.get("best_metric", "loss/total_val")  # runs saved before the key: the validation loss
    start, best = 1, float("inf")  # lowest best_metric so far
    if cfg["resume"]:
        start, best = resume(run_dir, model, opt, sched, device)
    run_dir, wandb_run = open_run(cfg, cfg["resume"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"run: {run_dir} | device: {device} | trainable params: {n_params}")

    model.train()
    sums, data_time, t0 = {}, 0.0, time.time()
    teacher_forcing = {**cfg, "loss": {**cfg["loss"], "rollout_steps": 1}}
    epoch_steps = len(train_loader)
    rollout_from = rollout_start(cfg["loss"], epoch_steps)
    if cfg["loss"]["rollout_steps"] > 1:
        print(f"k-step loss from step {rollout_from} (epoch {epoch_of(rollout_from, epoch_steps)}), {epoch_steps} steps per epoch")

    for step in range(start, cfg["iterations"] + 1):
        step_cfg = cfg if step >= rollout_from else teacher_forcing

        t = time.time()
        clip = next(train_iter).to(device, non_blocking=True)
        data_time += time.time() - t

        logging = step % cfg["log_every"] == 0
        if logging:  # the weight matrices before the step, for the update ratios
            before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad and p.ndim == 2}
        out = train_step(model, sigreg, clip, opt, sched, step_cfg, autocast)
        for k, v in out.items():
            sums[k] = sums.get(k, 0) + v.detach()

        if logging:
            n = cfg["log_every"]
            metrics = train_metrics(sums, n, cfg, model, before) | progress(
                sched, n, time.time() - t0, cfg["batch_size"], data_time, device
            ) | {"system/epoch": epoch_of(step, epoch_steps)}
            log(run_dir, step, metrics, wandb_run)
            sums, data_time, t0 = {}, 0.0, time.time()

        if step % cfg["eval_every"] == 0 or step == cfg["iterations"]:
            t_eval = time.time()
            if val_loader is not None:
                metrics = evaluate(model, sigreg, val_loader, step_cfg, device, autocast)
                metrics["prediction/tf_over_persistence_train"] = train_fit(model, train_probe, device, autocast)
                metrics |= gradient_shares(model, sigreg, train_probe[0].to(device), step_cfg, autocast)
                if hand_probe is not None:
                    metrics |= {f"probe/{k}": v for k, v in hand_probe.compute(model, device, autocast).items()}
                if motion_probe is not None:
                    metrics |= {f"motion_probe/{k}": v for k, v in motion_probe.compute(model, device, autocast).items()}
                log(run_dir, step, metrics, wandb_run)
                if metrics[best_metric] < best:
                    best = metrics[best_metric]
                    save_checkpoint(run_dir / "best.pt", model=trainable_state(model), step=step, **{best_metric: best})
            if cfg.get("save_every") and step % cfg["save_every"] == 0:  # to probe the run over training afterwards
                save_checkpoint(run_dir / f"step-{step}.pt", model=trainable_state(model), step=step)
            save_last(run_dir, model, opt, sched, step, best)
            t0 += time.time() - t_eval

    if wandb_run is not None:
        wandb_run.finish()


def train_metrics(sums, n, cfg, model, before):
    """The outputs of train_step summed over the last n steps -> the logged training metrics: loss terms (and their
    weighted share of the loss), gradient norms, share of clipped steps, update ratios (before: the weights before
    the last step)"""
    m = {k: v.item() / n for k, v in sums.items()}
    return {
        "loss/total": m["loss"],
        "loss/tf": m["tf"],
        "loss/kl": m["kl"],
        "loss/sigreg": m["sigreg"],
        "loss/kl_weighted": cfg["loss"]["beta"] * m["kl"],  # weighted terms: their real share of the loss
        "loss/sigreg_weighted": cfg["loss"]["sigreg"] * m["sigreg"],
        **{f"loss/{k}": v for k, v in m.items() if k.startswith("rollout")},
        **{f"optim/{k}": v for k, v in m.items() if k.startswith("gnorm")},
        "optim/clipped": m["clipped"],  # share of the steps whose gradient was clipped
        **{f"optim/{k}": v for k, v in update_ratios(model, before).items()},
    }


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


def layer_of(name):
    """Encoder or projector parameter -> its layer: embeddings, block_1..block_L, final_norm, projector"""
    if name.startswith("projector"):
        return "projector"
    block = re.search(r"\.layers?\.(\d+)\.", name)  # transformers 5: model.layers.i, 4: model.encoder.layer.i
    if block:
        return f"block_{int(block.group(1)) + 1}"
    return "embeddings" if "embeddings" in name else "final_norm"


def gradient_shares(model, sigreg, clip, cfg, autocast):
    """Which loss term moves the encoder and the projector, on one training batch (where.ipynb part 5: SIGReg gave 92 %
    of the encoder's gradient on the long CLS run, 100 % on patch16, tf opposed to it):
    - share_tf_<layer>: <g_pred, g> / |g|^2, g_pred the gradient of the prediction terms (tf and the rollout terms),
      g that of the whole loss; the shares of prediction, SIGReg and KL sum to 1, so SIGReg's is about 1 - this;
    - cos_tf_sigreg: cosine of the two gradients over the whole encoder (< 0: they pull against each other);
    - norm_tf, norm_sigreg: their norms over the whole encoder (weighted terms, before clipping).
    Adam rescales each parameter: the norms are not the update sizes, the shares give its direction."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and n.startswith(("encoder", "projector"))]
    with autocast:
        out = losses(model, sigreg, model.encode(clip), cfg)
    terms = {
        "tf": out["tf"] + sum(v for k, v in out.items() if k.startswith("rollout")),
        "sigreg": cfg["loss"]["sigreg"] * out["sigreg"],
        "kl": cfg["loss"]["beta"] * out["kl"],
    }
    grads = {}
    for i, (k, term) in enumerate(terms.items()):
        g = torch.autograd.grad(term, [p for _, p in named], retain_graph=i < len(terms) - 1, allow_unused=True)
        grads[k] = [torch.zeros_like(p).flatten() if gi is None else gi.flatten() for gi, (_, p) in zip(g, named)]
    layers = {}
    for i, (n, _) in enumerate(named):
        layers.setdefault(layer_of(n), []).append(i)
    if any(n.startswith("encoder") for n, _ in named):
        layers["encoder"] = [i for i, (n, _) in enumerate(named) if n.startswith("encoder")]
    metrics = {}
    for layer, idx in layers.items():
        g = {k: torch.cat([v[i] for i in idx]).float() for k, v in grads.items()}
        total = sum(g.values())
        metrics[f"grads/share_tf_{layer}"] = float(g["tf"] @ total / total.square().sum().clamp_min(1e-30))
    if "encoder" in layers:
        g = {k: torch.cat([v[i] for i in layers["encoder"]]).float() for k, v in grads.items()}
        metrics["grads/cos_tf_sigreg"] = float(F.cosine_similarity(g["tf"], g["sigreg"], dim=0))
        metrics["grads/norm_tf"], metrics["grads/norm_sigreg"] = float(g["tf"].norm()), float(g["sigreg"].norm())
    return metrics


def train_step(model, sigreg, clip, opt, sched, cfg, autocast):
    """One optimizer step on a batch of clips -> the loss terms and the gradient norms (optimizer_step)"""
    with autocast:
        out = losses(model, sigreg, model.encode(clip), cfg)
    return out | optimizer_step(out["loss"], model, opt, sched, cfg["optim"]["grad_clip"])


def losses(model, sigreg, emb, cfg):
    beta = cfg["loss"]["beta"]
    sig_w = cfg["loss"]["sigreg"]
    z, kl = model.idm(emb)  # (B, T-1, z_dim), one latent action per transition

    pred = model.predict(emb[:, :-1], z)  # (B, T-1, D)

    distance = F.l1_loss if cfg["loss"].get("prediction") == "l1" else F.mse_loss
    tf = distance(pred.float(), emb[:, 1:].float())  # every transition of the clip
    rollout = rollout_terms(model, emb, z, cfg["loss"]["rollout_steps"], cfg["loss"]["rollout_window"], distance)

    sig = sigreg(rearrange(emb.float(), "b t d -> t b d"))  # per frame across clips

    loss = tf + sum(rollout) + (beta * kl) + (sig_w * sig)

    return {"loss": loss, "tf": tf, "kl": kl, "sigreg": sig} | {
        f"rollout_{k}": term for k, term in enumerate(rollout, start=2)
    }


def rollout_terms(model, emb, z, steps, w, distance=F.mse_loss):
    """The k-step loss of JEPA-WM (Terver et al. 2025): L_k = error on a state predicted after k predictor steps from
    real states, for k = 2..steps (L_1 is tf). Each step reads the previous predictions detached (last-gradient only)
    in a window of w states (W_t), as at planning time; every state of the clip is a target once per k, so every
    initial context length is used (the paper samples one at random).
    emb: (B, T, D), z: (B, T-1, z_dim) the latent actions of the true transitions, distance: mse or l1, as tf
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
        )  # (B, T-k, D)
        if k > 1:
            terms.append(distance(pred.float(), emb[:, k:].float()))
        depth.append(
            torch.cat([emb[:, :k], pred], dim=1).detach()
        )  # positions < k never read at depth k
    return terms
