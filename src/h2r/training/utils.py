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

    return wandb.init(project=cfg["wandb"]["project"], name=run_dir.name, id=run_dir.name,
                      resume="allow", config=cfg, dir=run_dir)


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
    module (cs231n: about -3; far below, the module no longer learns). before: the matrices before the step"""
    ratios = {}
    for name, p in model.named_parameters():
        if name in before and before[name].float().std() > 0:
            r = (p.detach() - before[name]).float().std() / before[name].float().std()
            ratios.setdefault(name.split(".")[0], []).append(r.clamp_min(1e-12).log10().item())
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
