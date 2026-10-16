"""Does human video make the robot succeed more, or with fewer robot episodes? (h2r eval transfer)

Arms (same encoder, architecture, robot episodes per budget and evaluation; only the world model's video differs):
- A  robot only       IDM + world model trained on the n robot episodes
- B  human + robot    trained on SSv2, then fine-tuned on the n robot episodes
- C  human, frozen    trained on SSv2 only
The controller of every arm is trained on the same n robot episodes. Inputs: h2r eval pick runs, one per
(arm, n, training seed), all on the same test initial states. Computed again at every call, as runs are added.

- success S(n) per arm, 95% interval;
- gain S_arm(n) - S_A(n), paired: the initial states are resampled together for both arms, the training seeds
  within each arm;
- savings: the budgets of A between which S_A first reaches S_arm(n) (a bound, never extrapolated).
Robot data and training compute are reported separately: one does not stand for the other.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from h2r.training.utils import make_logger


ARMS = {"A": "A, robot only", "B": "B, human + robot", "C": "C, human frozen"}


def interval(x, y=None, draws=10000, seed=0):
    """Mean success of x, or gain mean(y) - mean(x), and its 95% bootstrap interval. x, y: (seeds, states) of 0/1 on
    the same initial states; states resampled jointly (paired), seeds within each arm"""
    rng = np.random.default_rng(seed)
    states = rng.integers(x.shape[1], size=(draws, x.shape[1]))

    def resample(r):
        seeds = rng.integers(len(r), size=(draws, len(r)))
        return r[seeds[:, :, None], states[:, None, :]].mean((1, 2))

    point, boot = (x.mean(), resample(x)) if y is None else (y.mean() - x.mean(), resample(y) - resample(x))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(point), [float(lo), float(hi)]


def savings(reference, target):
    """reference: {n: success} of A. The budgets of A between which it first reaches the target success:
    (previous budget, budget), (None, smallest) if already reached, (largest, None) if never"""
    budgets = sorted(reference)
    for i, n in enumerate(budgets):
        if reference[n] >= target:
            return (budgets[i - 1] if i else None, n)
    return (budgets[-1], None)


def load_successes(runs, policy):
    """{(arm, n): (seeds, states) array}, checking that every run was evaluated on the same initial states"""
    grouped, states = {}, None
    for run in runs:
        rows = json.load(open(run["path"]))["episodes"]
        names = [r["episode"] for r in rows]
        if states is None:
            states = names
        elif names != states:
            raise ValueError(f"{run['path']}: not the same test initial states as {runs[0]['path']}")
        grouped.setdefault((run["arm"], run["episodes"]), []).append([float(r[policy]["success"]) for r in rows])
    return {k: np.array(v) for k, v in grouped.items()}, states


def plot(table, path):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for arm, cells in table.items():
        ns = sorted(cells)
        mean = np.array([cells[n]["success"] for n in ns])
        lo, hi = np.array([cells[n]["ci"] for n in ns]).T
        ax.errorbar(ns, mean, yerr=[mean - lo, hi - mean], marker="o", lw=2, capsize=3, label=ARMS.get(arm, arm))
    ax.set_xscale("log")
    budgets = sorted({n for cells in table.values() for n in cells})
    ax.set_xticks(budgets, [str(n) for n in budgets])
    ax.minorticks_off()
    ax.set_xlabel("robot episodes")
    ax.set_ylabel("pick success")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run(cfg):
    run_dir = Path(cfg["output_dir"]) / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(run_dir)
    runs = [{"arm": arm, "episodes": n, "seed": seed,
             "path": str(Path(cfg["output_dir"]) / cfg["pick_runs"].format(arm=arm, n=n, seed=seed) / "results.json")}
            for arm in cfg["arms"] for n in cfg["budgets"] for seed in cfg["seeds"]]
    success, states = load_successes(runs, cfg["policy"])
    ref = cfg["reference"]
    table = {}
    for (arm, n), x in sorted(success.items()):
        point, ci = interval(x, draws=cfg["bootstrap"])
        table.setdefault(arm, {})[n] = {"success": point, "ci": ci, "seeds": len(x)}
    gains = {}
    for (arm, n), y in sorted(success.items()):
        if arm == ref or (ref, n) not in success:
            continue
        point, ci = interval(success[ref, n], y, draws=cfg["bootstrap"])
        bound = savings({m: c["success"] for m, c in table[ref].items()}, table[arm][n]["success"])
        gains.setdefault(arm, {})[n] = {"gain": point, "ci": ci, "reference_needs_between": bound}
    plot(table, run_dir / "transfer.png")
    json.dump({"success": table, "gain_over_reference": gains, "states": len(states), "compute": cfg.get("compute")},
              open(run_dir / "results.json", "w"), indent=2)

    log(f"{len(states)} test initial states | reference {ref}")
    for arm, cells in table.items():
        for n, c in sorted(cells.items()):
            line = f"{arm} n={n:<5} success {c['success']:.2f} [{c['ci'][0]:.2f}, {c['ci'][1]:.2f}] ({c['seeds']} seeds)"
            if n in gains.get(arm, {}):
                g = gains[arm][n]
                lo, hi = g["reference_needs_between"]
                line += (f" | gain {g['gain']:+.2f} [{g['ci'][0]:+.2f}, {g['ci'][1]:+.2f}] | {ref} reaches it with "
                         + (f"≤ {hi}" if lo is None else f"> {lo}" if hi is None else f"{lo}–{hi}") + " episodes")
            log(line)
    log(f"figure: {Path(run_dir) / 'transfer.png'}")
