"""Sanity check of the world model before a training run (h2r check wm): the model at initialization, then the
first optimizer steps with hooks on its modules. Report in <output_dir>/check-<name>-<time>/check.json"""

import contextlib
import json
import math
import re

import torch
from einops import rearrange

from h2r.models.jepa import build_model
from h2r.models.module import SIGReg, rope
from h2r.training.utils import (
    forever,
    grad_norms,
    make_autocast,
    make_run_dir,
    pick_device,
    seed_everything,
)
from h2r.training.world_model import (
    build_loaders,
    build_optimizers,
    losses,
    train_step,
)

SAMPLES = 16  # clips (or images) of the batch used for the attention statistics


def first_tensor(out):
    """The main tensor of a module output (Hugging Face outputs and tuples hold several)"""
    if torch.is_tensor(out):
        return out
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return out[0]


def attention_stats(q, k, mask=None):
    """q, k: (B, heads, L, d). Entropy of each query's attention over the log of the number of keys it sees
    (1 = uniform, 0 = a single key), averaged per head; largest logit"""
    logits = q.float() @ k.float().transpose(-1, -2) / math.sqrt(q.size(-1))
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    p = logits.softmax(-1)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)  # (B, heads, L)
    keys = mask.sum(-1).log() if mask is not None else math.log(k.size(-2))
    per_head = (entropy / keys).mean(dim=(0, 2))
    return {
        "entropy": per_head.mean().item(),
        "entropy_min_head": per_head.min().item(),
        "max_logit": logits[logits.isfinite()].max().item(),
    }


class Tracer:
    """Forward hooks for one training step: AdaLN gates of the predictor, norm of the residual stream after
    each block, attention entropy and largest logit of each self-attention"""

    def __init__(self, model):
        self.stats = {"gates": {}, "residual": {}, "attention": {}}
        self.handles, self.qk = [], {}
        for name, m in model.named_modules():
            kind = type(m).__name__
            if kind in ("Block", "ConditionalBlock", "ViTLayer"):
                self.hook(m, self.residual(name))
            if kind == "ConditionalBlock":
                self.hook(m.adaLN_modulation, self.gates(name))
            if kind == "Attention":
                self.hook(m, self.attention(name), with_kwargs=True)
            if kind == "ViTAttention":
                heads = model.encoder.model.config.num_attention_heads
                self.hook(m.q_proj, self.keep(name, "q", heads))
                self.hook(m.k_proj, self.keep(name, "k", heads))
                self.hook(m, self.vit_attention(name))

    def hook(self, module, fn, **kwargs):
        """The statistics stay out of the autograd graph of the training step"""
        self.handles.append(module.register_forward_hook(torch.no_grad()(fn), **kwargs))

    def residual(self, name):
        def fn(m, args, out):
            self.stats["residual"][name] = first_tensor(out).float().norm(dim=-1).mean().item()

        return fn

    def gates(self, name):
        def fn(m, args, out):
            gate_msa, gate_mlp = out.chunk(6, dim=-1)[2::3]
            self.stats["gates"][name] = {
                "msa": gate_msa.float().abs().mean().item(),
                "mlp": gate_mlp.float().abs().mean().item(),
            }

        return fn

    def attention(self, name):
        def fn(m, args, kwargs, out):  # recomputes the logits of Attention.forward
            x = m.norm(args[0][:SAMPLES])
            q, k, _ = (
                rearrange(t, "b t (h d) -> b h t d", h=m.heads)
                for t in m.to_qkv(x).chunk(3, dim=-1)
            )
            frame = torch.arange(x.size(1), device=x.device) // kwargs.get("frame_size", 1)
            q, k = rope(q, frame), rope(k, frame)
            mask = frame[:, None] >= frame[None, :] if kwargs.get("causal", True) else None
            self.stats["attention"][name] = attention_stats(q, k, mask)

        return fn

    def keep(self, name, which, heads):
        def fn(m, args, out):
            self.qk[name, which] = rearrange(out[:SAMPLES], "b t (h d) -> b h t d", h=heads)

        return fn

    def vit_attention(self, name):
        def fn(m, args, out):
            self.stats["attention"][name] = attention_stats(self.qk.pop((name, "q")), self.qk.pop((name, "k")))

        return fn

    def close(self):
        for h in self.handles:
            h.remove()
        return self.stats


def seeded_losses(model, sigreg, clip, cfg, precision):
    """Losses on one batch with the generators reset, so two calls draw the same noise"""
    seed_everything(cfg["seed"])
    with precision:
        out = losses(model, sigreg, model.encode(clip), cfg)
    return {k: v.item() for k, v in out.items()}


def output_stats(model, sigreg, clip, cfg, autocast):
    """Mean and std of the output of each top-level module, on one forward pass"""
    stats, handles = {}, []
    for name, module in model.named_children():
        def fn(m, args, out, name=name):
            x = first_tensor(out).detach().float()
            stats[name] = {"mean": x.mean().item(), "std": x.std().item()}

        handles.append(module.register_forward_hook(fn))
    with autocast:  # with gradients: on MPS, attention dropout has no no_grad kernel
        emb = model.encode(clip)
        losses(model, sigreg, emb, cfg)
    for h in handles:
        h.remove()
    emb = emb.float()
    stats["targets"] = {"var": emb[:, 1:].var().item(), "persistence": (emb[:, 1:] - emb[:, :-1]).pow(2).mean().item()}
    return stats


def zero_grads(model):
    """Trainable parameters whose gradient is exactly zero, block indices replaced by *, with their count"""
    names = [
        re.sub(r"\.\d+\.", ".*.", n)
        for n, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or not p.grad.any())
    ]
    return {n: names.count(n) for n in dict.fromkeys(names)}


def at_init(cfg, model, sigreg, clip, device, autocast):
    report = {}
    seed_everything(cfg["seed"])
    twin = build_model(cfg["model"]).to(device)
    report["same_init"] = all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), twin.state_dict().values()))
    del twin

    bf16 = seeded_losses(model, sigreg, clip, cfg, autocast)
    report["losses"] = bf16
    report["same_losses"] = bf16 == seeded_losses(model, sigreg, clip, cfg, autocast)
    fp32 = seeded_losses(model, sigreg, clip, cfg, contextlib.nullcontext())
    report["fp32_relative_diff"] = {k: abs(bf16[k] - fp32[k]) / max(abs(fp32[k]), 1e-12) for k in fp32}

    report["outputs"] = output_stats(model, sigreg, clip, cfg, autocast)

    seed_everything(cfg["seed"])
    with autocast:
        out = losses(model, sigreg, model.encode(clip), cfg)
    out["loss"].backward()
    report["grad_norms"] = {k: v.item() for k, v in grad_norms(model).items()}
    report["zero_grads"] = zero_grads(model)
    model.zero_grad(set_to_none=True)
    return report


def traced_steps(n):
    """1, 2, 3, 5, 10, 20, 50, 100, ... up to n, and n"""
    steps = {s * 10**e for e in range(6) for s in (1, 2, 3, 5)}
    return sorted(s for s in steps if s <= n) + ([n] if n not in steps else [])


def first_steps(cfg, model, sigreg, loader, device, autocast):
    """check_steps optimizer steps on the training batches: losses, gradient norms and clipping, update size per
    module and optimizer at every step; the hooks of Tracer at the traced steps"""
    params = [p for p in model.parameters() if p.requires_grad]
    opts, scheds = build_optimizers(model, cfg)
    optimizer = {id(p): label for o, label in zip(opts, ("muon", "adamw")) for g in o.param_groups for p in g["params"]}
    groups = {}
    for module, child in model.named_children():
        for p in child.parameters():
            if id(p) in optimizer:
                groups.setdefault(f"{module}/{optimizer[id(p)]}", []).append(p)

    traced, rows, batches = traced_steps(cfg["check_steps"]), [], forever(loader)
    for step in range(1, cfg["check_steps"] + 1):
        clip = next(batches).to(device, non_blocking=True)
        before = {k: [p.detach().clone() for p in ps] for k, ps in groups.items()}
        tracer = Tracer(model) if step in traced else None
        out = train_step(model, sigreg, clip, opts, scheds, params, cfg, autocast)
        row = {"step": step} | {k: v.item() for k, v in out.items()}
        row["clipped"] = row["gnorm"] > cfg["optim"]["grad_clip"]
        row["update_ratio"] = {
            k: (sum((p.detach() - b).pow(2).sum() for p, b in zip(ps, before[k])).sqrt()
                / sum(b.pow(2).sum() for b in before[k]).sqrt()).item()
            for k, ps in groups.items()
        }
        if tracer is not None:
            row |= tracer.close()
        rows.append(row)
    return rows


def flags(init, rows, modules):
    """Pass / fail of the checks that have an expected answer; modules: the trainable top-level modules"""
    finite = all(math.isfinite(r["loss"]) for r in rows)
    with_gradient = {k for r in rows for k, v in r.items() if k.startswith("gnorm_") and v > 0}
    return {
        "same_init_for_the_same_seed": init["same_init"],
        "same_losses_for_the_same_seed": init["same_losses"],
        "kl_zero_at_init": init["losses"]["kl"] == 0,
        "bf16_within_5pct_of_fp32": all(v < 0.05 for v in init["fp32_relative_diff"].values()),
        "finite_losses": finite,
        "every_module_gets_gradient": with_gradient == {f"gnorm_{m}" for m in modules},
    }


def check(cfg):
    seed_everything(cfg["seed"])
    device = pick_device()
    autocast = make_autocast(cfg["precision"], device)
    model = build_model(cfg["model"]).to(device)
    sigreg = SIGReg().to(device)
    loader, _ = build_loaders(cfg, device)
    clip = next(iter(loader)).to(device)
    run_dir = make_run_dir(cfg | {"name": f"check-{cfg['name']}"})

    init = at_init(cfg, model, sigreg, clip, device, autocast)
    rows = first_steps(cfg, model, sigreg, loader, device, autocast)
    modules = [n for n, c in model.named_children() if any(p.requires_grad for p in c.parameters())]
    report = {"flags": flags(init, rows, modules), "init": init, "steps": rows}
    (run_dir / "check.json").write_text(json.dumps(report, indent=1))

    print(f"check: {run_dir / 'check.json'} | device: {device}")
    for k, v in report["flags"].items():
        print(f"{'ok  ' if v else 'FAIL'} {k}")
    print("zero gradients at step 0:", init["zero_grads"])
    for r in rows:
        if "gates" in r:
            gates = [g["msa"] for g in r["gates"].values()]
            print(
                f"step {r['step']} | tf {r['tf']:.4g} | gnorm {r['gnorm']:.3g} | idm {r.get('gnorm_idm', 0):.3g}"
                f" | gates {min(gates):.3g}-{max(gates):.3g}"
                f" | update encoder/muon {r['update_ratio'].get('encoder/muon', 0):.2g}"
            )
