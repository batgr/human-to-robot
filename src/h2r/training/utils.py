import torch
import random
import numpy as np
import contextlib
import subprocess
from datetime import datetime
from pathlib import Path
import yaml
import math
import json
import os


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def pick_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )


def make_autocast(precision, device):
    """bf16 or fp16 mixed precision, or plain fp32"""
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(device.type, dtype=dtype)


def git_info():
    """Commit of the code"""
    run = lambda *args: subprocess.check_output(["git", *args], text=True).strip()
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def make_run_dir(cfg):
    run_dir = Path(cfg["output_dir"]) / f"{cfg['name']}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    git = git_info()
    if git["dirty"]:
        diff = subprocess.check_output(["git", "diff", "HEAD"], text=True)
        (run_dir / "diff.patch").write_text(diff)
    yaml.safe_dump(
        cfg | {"git": git}, open(run_dir / "config.yaml", "w"), sort_keys=False
    )
    return run_dir


def init_wandb(cfg, run_dir):
    """W&B run named after the run directory, None when disabled"""
    if not cfg["wandb"]["enabled"]:
        return None
    import wandb

    return wandb.init(
        project=cfg["wandb"]["project"],
        name=run_dir.name,
        id=run_dir.name,
        resume="allow",
        config=cfg,
        dir=run_dir,
    )


def forever(loader):
    """restarts the loader at the end of each epoch"""
    while True:
        yield from loader


def warmup_cosine(warmup, total):

    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))

    return f


def build_optimizer(model, optim, iterations):
    """AdamW (no weight decay on biases, norms, embeddings and learned queries) and its warmup-cosine schedule;
    optim: lr, weight_decay, warmup (share of the iterations)"""
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
    schedule = warmup_cosine(int(optim["warmup"] * iterations), iterations)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, schedule)


def save_figures(figs, run_dir, step, wandb_run=None, prefix="figures"):
    """{name: matplotlib figure} -> name.png in the run directory, and W&B images under prefix/; the figures are
    closed"""
    from matplotlib import pyplot as plt

    for name, fig in figs.items():
        fig.savefig(Path(run_dir) / f"{name}.png", dpi=120)
    if wandb_run is not None:
        import wandb

        wandb_run.log(
            {f"{prefix}/{k}": wandb.Image(f) for k, f in figs.items()}, step=step
        )
    for fig in figs.values():
        plt.close(fig)


def peak_memory_gb(device):
    """Peak memory held by tensors on the GPU, None on CPU"""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return None


def grad_norms(model):
    """Gradient norm of each top-level module (encoder, predictor, idm, ...), before clipping"""
    norms = {}
    for name, module in model.named_children():
        grads = [p.grad.norm() for p in module.parameters() if p.grad is not None]
        if grads:
            norms[f"gnorm_{name}"] = torch.stack(grads).norm()
    return norms


def update_ratios(model, before):
    """log10 of std(update) / std(weight) for one optimizer step, median over the weight matrices of each top-level
    module (cs231n: about -3; far below, the module no longer learns). before: the matrices before the step
    """
    ratios = {}
    for name, p in model.named_parameters():
        if name in before and before[name].float().std() > 0:
            r = (p.detach() - before[name]).float().std() / before[name].float().std()
            ratios.setdefault(name.split(".")[0], []).append(
                r.clamp_min(1e-12).log10().item()
            )
    return {f"update_ratio_{k}": float(np.median(v)) for k, v in ratios.items()}


def log(run_dir, step, metrics, wandb_run=None):
    """One line in run_dir/metrics.jsonl, the same metrics to W&B if enabled (one section per category,
    the prefix before "/"), and every metric printed, one line per section"""
    metrics = {k: round(v, 6) for k, v in metrics.items() if v is not None}
    with open(run_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps({"step": step} | metrics) + "\n")
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)
    sections = {}
    for key, value in metrics.items():
        section, _, name = key.partition("/")
        sections.setdefault(section, []).append(f"{name} {value:.4g}")
    print(f"step {step}")
    for section, items in sections.items():
        print(f"  {section:<15}" + " | ".join(items))


def rng_state():
    """States of every random generator, to resume a run exactly"""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": torch.mps.get_rng_state() if torch.mps.is_available() else None,
    }


def set_rng_state(state):
    """The generator states are CPU tensors, even in a checkpoint loaded onto the GPU"""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])
    if state["mps"] is not None:
        torch.mps.set_rng_state(state["mps"].cpu())


def save_checkpoint(path, **state):
    """Written to a temporary file then renamed: a crash never leaves a half-written checkpoint"""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def setup(cfg):
    """Seeded generators, the device and the autocast context of a run"""
    seed_everything(cfg["seed"])
    device = pick_device()
    return device, make_autocast(cfg["precision"], device)


def open_run(cfg, run_dir=None):
    """A new run directory (config.yaml, git commit and diff), or run_dir when resuming, and its W&B run"""
    run_dir = Path(run_dir) if run_dir else make_run_dir(cfg)
    return run_dir, init_wandb(cfg, run_dir)


def trainable_state(model):
    """State dict without the frozen weights (a frozen encoder reloads from the Hub), buffers kept"""
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k not in frozen}


def epoch_of(step, epoch_steps):
    """0, 1, 2...: the epoch of a step (steps start at 1, epoch e = steps e * epoch_steps + 1 to (e + 1) * epoch_steps)"""
    return max(step - 1, 0) // epoch_steps


def optimizer_step(loss, model, opt, sched, grad_clip=None):
    """Backward, gradient norm of each top-level module, clipping of the whole gradient when grad_clip is set, optimizer
    and schedule steps -> {gnorm_<module>, and with grad_clip: gnorm (before clipping), clipped (1 when clipped)}"""
    opt.zero_grad(set_to_none=True)
    loss.backward()
    out = grad_norms(model)
    if grad_clip is not None:
        params = [p for p in model.parameters() if p.requires_grad]
        out["gnorm"] = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        out["clipped"] = (out["gnorm"] > grad_clip).float()
    opt.step()
    sched.step()
    return out


def progress(sched, n, elapsed, batch_size, data_time=None, device=None):
    """Learning rate and speed over the last n steps (elapsed, data_time: seconds)"""
    return {
        "optim/lr": sched.get_last_lr()[0],
        "system/clips_per_s": n * batch_size / elapsed,
        "system/step_s": elapsed / n,
        "system/data_s": None if data_time is None else data_time / n,  # waiting for the loader
        "system/mem_gb": None if device is None else peak_memory_gb(device),
    }


def save_last(run_dir, model, opt, sched, step, best):
    """last.pt: everything resume needs to continue the run exactly"""
    save_checkpoint(Path(run_dir) / "last.pt", model=trainable_state(model), opt=opt.state_dict(),
                    sched=sched.state_dict(), step=step, best=best, rng=rng_state())


def resume(run_dir, model, opt, sched, device):
    """Restores model, optimizer, schedule and random generators from run_dir/last.pt and drops the metrics logged
    after it (they will be logged again) -> the next step and the best value so far"""
    ckpt = torch.load(Path(run_dir) / "last.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)  # a frozen encoder is not in the checkpoint
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    set_rng_state(ckpt["rng"])
    start = ckpt["step"] + 1
    metrics = Path(run_dir) / "metrics.jsonl"
    lines = metrics.read_text().splitlines(keepends=True)
    metrics.write_text("".join(line for line in lines if json.loads(line)["step"] < start))
    return start, ckpt["best"]
