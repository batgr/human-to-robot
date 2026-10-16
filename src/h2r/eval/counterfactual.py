"""When the action changes, does the world model predict a different and coherent consequence? (h2r eval counterfactual)

From the frame t0 of each held-out test episode, k action sequences of `horizon` steps are drawn as the CEM draws
them, sent as commands and played in the simulator from the same restored state; the frame reached is encoded. The
world model predicts the final state of each sequence from the same actions:
- controller  the controller in the loop on the sequence (what the planner scores);
- oracle      the IDM's latent actions on the real frames (the world model alone).
Metrics, in the world model's space, over the k outcomes:
- identification  share of predictions closer to their own real outcome than to the k - 1 others (chance: 1 / k);
- direction       cosine between predicted and real differences of every pair of outcomes;
- spread          predicted over real distance between outcomes (0: the actions are ignored).
"""

import json

import numpy as np
import torch
import yaml

from h2r.eval.episodes import command_map, held_out_episodes, make_env, planner_context, play
from h2r.plan import encode, load, physical
from h2r.training.utils import make_logger, prepare_run, setup


def scores(pred, real):
    """pred, real: (k, F) predicted and real outcomes of k sequences"""
    identified = (torch.cdist(pred, real).argmin(1) == torch.arange(len(pred), device=pred.device)).float().mean()
    i, j = torch.triu_indices(len(pred), len(pred), 1)
    dp, dr = pred[i] - pred[j], real[i] - real[j]
    return {"identification": float(identified),
            "direction": float(torch.nn.functional.cosine_similarity(dp, dr, dim=-1).mean()),
            "spread": float((dp.norm(dim=-1) / dr.norm(dim=-1).clamp(min=1e-8)).median())}


@torch.no_grad()
def run(cfg):
    cfg, run_dir, _ = prepare_run(cfg | {"name": f"{cfg['name']}/counterfactual"})
    log = make_logger(run_dir)
    device, _ = setup(cfg)
    sim_cfg = yaml.safe_load(open(cfg["sim_config"]))
    model, ctrl = load(cfg["controller"], device)
    env, cmap, p, H = make_env(cfg, sim_cfg), command_map(cfg, sim_cfg), cfg["planner"], cfg["horizon"]
    k, gen = cfg["counterfactual"]["sequences"], torch.Generator().manual_seed(cfg["seed"])
    per_episode = {"controller": [], "oracle": []}
    for ep in held_out_episodes(cfg, sim_cfg):
        obs = env.restore(ep["sim"][ep["t0"]]["sim_state"])
        emb, latents = planner_context(model, ctrl, env, obs, ep, cfg, device)
        acts = physical(torch.randn(k, H, ctrl.act_mean.numel(), generator=gen).to(device), ctrl, p)
        real = torch.stack([encode(model, play(env, ep, a, cmap, sim_cfg["frameskip"], cfg["camera"])[1], device)
                            for a in acts.cpu().numpy()])  # (k, H, N, D)
        z = model.idm(torch.cat([emb[-1:].expand(k, 1, *emb.shape[1:]), real], 1))[0]  # (k, H, d_a)
        pred = {"controller": model.rollout(emb[None], latents[None], acts[None], ctrl, p["window"])[0, :, -1],
                "oracle": model.rollout(emb[None], latents[None], z[None], None, p["window"])[0, :, -1]}
        for level, final in pred.items():
            per_episode[level].append(scores(final.flatten(1).float(), real[:, -1].flatten(1).float()))
        log(f"{ep['name']} | " + " | ".join(
            f"{lv} " + " ".join(f"{m} {v:.2f}" for m, v in per_episode[lv][-1].items()) for lv in per_episode))
    summary = {lv: {m: float(np.mean([s[m] for s in rows])) for m in rows[0]} for lv, rows in per_episode.items()}
    summary["chance_identification"] = 1 / k
    json.dump({"summary": summary, "episodes": per_episode}, open(run_dir / "results.json", "w"), indent=2)
    for lv in ("controller", "oracle"):
        log(f"{lv}: identification {summary[lv]['identification']:.2f} (chance {1 / k:.2f}) | direction "
            f"{summary[lv]['direction']:.2f} | spread {summary[lv]['spread']:.2f}")
