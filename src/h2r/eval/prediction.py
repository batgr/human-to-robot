"""Does the world model predict the robot's future? (h2r eval prediction)

On held-out test episodes, from every stride-th frame t, the predictor is unrolled h = 1..horizon steps:
- oracle      latent actions from the IDM, which sees the real future: the best the world model can do;
- controller  the controller on the recorded actions and the real states;
- loop        the controller in the loop, on the predicted states.
skill(h) = 1 - error / error of persistence (the current state repeated): 0 is no better than predicting no change.
Picture: the real future frames, and for each predicted state the nearest real frame of the other test episodes.
"""

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image

from h2r.data.prepare import extract_frames, split_of, states_to_actions
from h2r.models.controller import load_controller
from h2r.training.utils import make_logger, prepare_run, setup

LEVELS = ("oracle", "controller", "loop")


def test_episodes(cfg, sim_cfg):
    """Camera frames and robot states of the first cfg["episodes"] test episodes (split as h2r data extract does)"""
    path = f"{cfg['robot']}/{cfg['task']}/{cfg['episodes_from']}"
    out = []
    for folder in sorted((Path(sim_cfg["raw_dir"]) / path).glob("episode_*")):
        if len(out) == cfg["episodes"]:
            break
        if split_of(f"{path}/{folder.name}", sim_cfg["split"]) != "test":
            continue
        video = folder / f"{cfg['camera']}.mp4"
        frames = [np.asarray(Image.open(io.BytesIO(j))) for j in extract_frames(video, sim_cfg["fps"], sim_cfg["image_size"])]
        states = np.asarray(pq.read_table(folder / "state.parquet")["state"].to_pylist())
        out.append({"name": folder.name, "frames": frames, "states": states})
    if not out:
        raise ValueError(f"no test episode in {Path(sim_cfg['raw_dir']) / path}")
    return out


@torch.no_grad()
def encode(model, frames, device):
    """(T, H, W, 3) uint8 -> (T, N, D)"""
    return model.encode(torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].to(device))[0]


@torch.no_grad()
def predictions(model, ctrl, emb, actions, C, H, starts):
    """Predicted states (S, H, N, D) of the three levels from the starts t (context t - C + 1 .. t)"""
    z = model.idm(emb[None])[0][0]  # eval: mu, (T - 1, d_a)
    zc = ctrl(actions, emb[:-1])
    ctx = torch.stack([emb[t - C + 1 : t + 1] for t in starts])
    past = lambda lat: torch.stack([lat[t - C + 1 : t] for t in starts])  # noqa: E731
    future = lambda seq: torch.stack([seq[t : t + H] for t in starts])[:, None]  # noqa: E731
    return {
        "oracle": model.rollout(ctx, past(z), future(z), None, C)[:, 0],
        "controller": model.rollout(ctx, past(zc), future(zc), None, C)[:, 0],
        "loop": model.rollout(ctx, past(zc), future(actions), ctrl, C)[:, 0],
    }


def plot(skill, H, path):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.axhline(0, color="0.6", lw=1, ls="--", label="persistence")
    for lv in LEVELS:
        ax.plot(range(1, H + 1), skill[lv], marker="o", ms=4, lw=2, label=lv)
    ax.set_xlabel("horizon h (steps of 0.25 s)")
    ax.set_ylabel("skill = 1 − err / persistence")
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_imagined(real, imagined, path):
    """Two rows: the real frames t + 1..t + H, the nearest real frames of the predicted states"""
    fig, axes = plt.subplots(2, len(real), figsize=(1.8 * len(real), 3.8))
    for h in range(len(real)):
        for row, (img, label) in enumerate(((real[h], "real"), (imagined[h], "predicted (nearest frame)"))):
            axes[row, h].imshow(img)
            axes[row, h].axis("off")
            axes[row, h].set_title(f"{label}, h = {h + 1}" if h == 0 else f"h = {h + 1}", fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(cfg):
    cfg, run_dir, done = prepare_run(cfg | {"name": f"{cfg['name']}/prediction"})
    if done:
        print(f"already evaluated: {run_dir}")
        return
    log = make_logger(run_dir)
    device, _ = setup(cfg)
    sim_cfg = yaml.safe_load(open(cfg["sim_config"]))
    model, ctrl = load_controller(cfg["controller"], device)
    C, H = cfg["context"], cfg["horizon"]
    err, persist, n = {lv: np.zeros(H) for lv in LEVELS}, np.zeros(H), 0
    shown = None
    episodes = test_episodes(cfg, sim_cfg)
    for ep in episodes:
        ep["emb"] = encode(model, ep["frames"], device)
        actions = torch.tensor(states_to_actions(ep["states"]), dtype=torch.float32, device=device)
        starts = list(range(C - 1, len(ep["emb"]) - H, cfg["stride"]))
        preds = predictions(model, ctrl, ep["emb"], actions, C, H, starts)
        target = torch.stack([ep["emb"][t + 1 : t + H + 1] for t in starts])
        now = torch.stack([ep["emb"][t] for t in starts])[:, None]
        persist += (now - target).pow(2).sum((-2, -1)).sum(0).cpu().numpy()
        for lv, p in preds.items():
            err[lv] += (p - target).pow(2).sum((-2, -1)).sum(0).cpu().numpy()
        n += len(starts)
        shown = shown or (ep, starts[len(starts) // 2], preds["loop"][len(starts) // 2])
    skill = {lv: (1 - err[lv] / persist).tolist() for lv in LEVELS}
    ratio = (err["loop"] / err["oracle"]).tolist()
    plot(skill, H, run_dir / "skill.png")

    ep, t, pred = shown  # nearest frames among the other episodes
    pool = [(e["emb"][i], e["frames"][i]) for e in episodes if e is not ep for i in range(len(e["emb"]))]
    if pool:
        bank = torch.stack([s for s, _ in pool]).flatten(1)
        nearest = torch.cdist(pred.flatten(1).float(), bank.float()).argmin(1).tolist()
        plot_imagined(ep["frames"][t + 1 : t + H + 1], [pool[i][1] for i in nearest], run_dir / "imagined.png")

    json.dump({"skill": skill, "loop_to_oracle": ratio, "predictions": n}, open(run_dir / "results.json", "w"), indent=2)
    for lv in LEVELS:
        log(f"{lv:10} skill h1..h{H} " + " ".join(f"{s:.2f}" for s in skill[lv]))
    log(f"loop / oracle error h1..h{H} " + " ".join(f"{r:.2f}" for r in ratio) + f" | {n} starts, {len(episodes)} episodes")
