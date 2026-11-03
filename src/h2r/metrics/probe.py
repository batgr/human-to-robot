"""Linear probes during training: does the state (and the encoder) keep where the hand is and the direction of the clip
(HandProbe, SSv2), where the robot's end effector is and how it moves (RobotProbe, DROID / sim)? The protocol of
where.ipynb: tokens pooled on a 4 x 4 grid, a PCA shared by the regions, ridge held out by video (2 folds by video
parity, lambda chosen on an inner split by video)"""

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from h2r.data.dataset import decode, shards

DIRECTIONS = {  # SSv2 classes whose label gives the direction of the motion in the image
    "left_right": {
        "Pushing something from left to right": 1,
        "Pulling something from left to right": 1,
        "Pushing something from right to left": -1,
        "Pulling something from right to left": -1,
    },
    "up_down": {"Moving something up": 1, "Moving something down": -1},
}
LAMBDAS = (1e-3, 1e-2, 1e-1, 1, 10, 100)


def grid(x):
    """(n, N, d) tokens of a square grid -> (n, 16 * d) pooled 4 x 4; (n, d) a single vector, kept"""
    if x.dim() == 2:
        return x
    n, N, d = x.shape
    side = int(N**0.5)
    return F.adaptive_avg_pool2d(x.reshape(n, side, side, d).permute(0, 3, 1, 2), min(side, 4)).flatten(1)


def pca(x, fit, k=24):
    """x (n, R * d): R regions of d dims (16 for a pooled grid) -> (n, R * k), PCA shared by the regions, fitted on
    the rows fit"""
    regions = 16 if x.shape[1] % 16 == 0 and x.shape[1] // 16 > k else 1
    x = x.view(len(x), regions, -1)
    rows = x[fit].reshape(-1, x.shape[-1])
    mean = rows.mean(0)
    rows = rows[torch.randperm(len(rows), generator=torch.Generator().manual_seed(0))[:50000]] - mean
    basis = torch.linalg.svd(rows, full_matrices=False)[2][:k].T
    return ((x - mean) @ basis).flatten(1)


def ridge(x, y, lam):
    return torch.linalg.solve(x.T @ x + lam * len(x) * torch.eye(x.shape[1], dtype=x.dtype), x.T @ y)


def heldout(x, y, vid):
    """Held-out linear prediction of every row: 2 folds by video parity, x standardized on the fit fold"""
    pred = torch.zeros_like(y)
    for fold in (0, 1):
        fit = torch.as_tensor(vid % 2 == fold)
        mean, std = x[fit].mean(0), x[fit].std(0) + 1e-6
        xs = (x - mean) / std
        a, ym = xs[fit], y[fit].mean(0)
        inner = torch.as_tensor((vid[fit.numpy()] // 2) % 2 == 0)
        lam = 1.0 if inner.all() or not inner.any() else min(LAMBDAS, key=lambda l: ((a[~inner] @ ridge(a[inner], y[fit][inner] - ym, l) - (y[fit][~inner] - ym)) ** 2).sum().item())
        pred[~fit] = xs[~fit] @ ridge(a, y[fit] - ym, lam) + ym
    return pred


@torch.no_grad()
def frame_features(model, videos, need, device, autocast):
    """{"state", "encoder"}: (n, F) features of the frames need {video index: frame indices} (the state, and the
    encoder's last patch tokens, pooled 4 x 4), and the row of each (video, frame)"""
    was_training = model.training
    model.eval()
    feats, keys = {"state": [], "encoder": []}, []
    for v, ts in need.items():
        px = decode(videos[v]["frames"])[ts].to(device)
        with autocast:
            feats["state"].append(grid(model.encode(px[None])[0]).float().cpu())
            tokens = model.encoder(px.float(), interpolate_pos_encoding=True).last_hidden_state
            feats["encoder"].append(grid(tokens[:, 1:]).float().cpu())
        keys += [(v, t) for t in ts]
    model.train(was_training)
    return {k: torch.cat(f).double() for k, f in feats.items()}, {k: i for i, k in enumerate(keys)}


def r2(pred, y):
    """R² over every dimension of y (pooled sums of squares)"""
    return float(1 - (pred - y).pow(2).sum() / (y - y.mean(0)).pow(2).sum())


class HandProbe:
    """
    - hand_r2_{state, encoder}: R² of the tracked hand's position (x, y in the image) from the world model's state and
      from the encoder's last patch tokens, on up to `frames` validation frames with a hand (where.ipynb: DINOv2-B
      patches 0.42, our encoders 0.00-0.07, an untrained one 0.04);
    - left_right_{state, encoder}, up_down_{state, encoder}: accuracy of the clip's direction (SSv2 label) from
      [mean over the clip, last - first] (DINOv2-B 0.86 / 0.83, ours 0.52-0.66; chance 0.5).
    hands: parquet of video, x, y (one value per frame of the video, NaN where no hand is tracked)
    """

    def __init__(self, root, hands, frames=3000, seed=0):
        table = pq.read_table(hands).to_pylist()
        hand = {r["video"]: np.stack([r["x"], r["y"]], 1) for r in table}
        labels = dict(zip(*[pq.read_table(f"{root}/labels.parquet").column(c).to_pylist() for c in ["video", "label"]]))
        self.videos = [r for p in shards(root, "val") for r in pq.read_table(p, columns=["video", "frames"]).to_pylist()
                       if r["video"] in hand]
        seen = [(v, t) for v, r in enumerate(self.videos) for t, xy in enumerate(hand[r["video"]]) if not np.isnan(xy[0])]
        pick = np.random.default_rng(seed).choice(len(seen), min(frames, len(seen)), replace=False)
        self.hand_rows = [seen[i] for i in sorted(pick)]
        self.target = torch.tensor(np.array([hand[self.videos[v]["video"]][t] for v, t in self.hand_rows]), dtype=torch.float64)
        self.clips = {task: [(v, m[labels[r["video"]]]) for v, r in enumerate(self.videos) if labels.get(r["video"]) in m]
                      for task, m in DIRECTIONS.items()}
        need = {}
        for v, t in self.hand_rows:
            need.setdefault(v, set()).add(t)
        for clips in self.clips.values():
            for v, _ in clips:
                need.setdefault(v, set()).update(range(len(self.videos[v]["frames"])))
        self.need = {v: sorted(ts) for v, ts in need.items()}

    def compute(self, model, device, autocast):
        feats, row = frame_features(model, self.videos, self.need, device, autocast)
        hand_idx = [row[k] for k in self.hand_rows]
        vid = np.array([v for v, _ in self.hand_rows])
        fit = torch.zeros(len(row), dtype=torch.bool)
        fit[hand_idx] = True
        out = {}
        for name, f in feats.items():
            x = pca(f, fit)
            pred = heldout(x[hand_idx], self.target, vid)
            out[f"hand_r2_{name}"] = r2(pred, self.target)
            for task, clips in self.clips.items():
                if len(clips) < 20:  # too few labelled clips in the split
                    continue
                xc = torch.stack([
                    torch.cat([x[[row[(v, t)] for t in range(len(self.videos[v]["frames"]))]].mean(0),
                               x[row[(v, len(self.videos[v]["frames"]) - 1)]] - x[row[(v, 0)]]])
                    for v, _ in clips])
                yc = torch.tensor([[d] for _, d in clips], dtype=torch.float64)
                pc = heldout(xc, yc, np.array([v for v, _ in clips]))
                out[f"{task}_{name}"] = float((torch.sign(pc) == yc).double().mean())
        return out


class RobotProbe:
    """The end effector read from the states of an action-conditioned or latent-action model (DROID, sim), on up to
    `frames` validation frames t of episodes with states, each with the next frame t + 1, held out by episode:
    - ee_xyz_r2_{state, encoder}: R² of the end-effector position x, y, z from the frame's features. DROID: in the
      robot's base frame while the camera moves from scene to scene (H0), so a hard target across episodes;
    - gripper_r2_{state, encoder}: R² of the gripper closedness, the same under any camera;
    - dxyz_r2_{state, encoder}, dgripper_r2_{state, encoder}: R² of the action of the transition (t, t + 1), its
      position delta and gripper delta, from [x_t, x_t+1 - x_t]: does the state show how the robot moves?
    LeWM reports Pearson r (Push-T agent 0.974 linear, OGBench-Cube block 0.997, gripper 0.938): r 0.97 is R² ~ 0.94.
    PCA of 64 dims per region (24 for the hand): the CLS state has 192, the gripper may sit in a small direction."""

    def __init__(self, root, frames=3000, seed=0):
        self.videos = [r for p in shards(root, "val")
                       for r in pq.read_table(p, columns=["video", "frames", "actions", "states"]).to_pylist()
                       if r["states"] is not None and r["actions"] is not None]
        pairs = [(v, t) for v, r in enumerate(self.videos) for t in range(len(r["frames"]) - 1)]
        pick = np.random.default_rng(seed).choice(len(pairs), min(frames, len(pairs)), replace=False)
        self.rows = [pairs[i] for i in sorted(pick)]
        states = torch.tensor(np.array([self.videos[v]["states"][t] for v, t in self.rows]), dtype=torch.float64)
        actions = torch.tensor(np.array([self.videos[v]["actions"][t] for v, t in self.rows]), dtype=torch.float64)
        self.targets = {"ee_xyz": states[:, :3], "gripper": states[:, 6:], "dxyz": actions[:, :3], "dgripper": actions[:, 6:]}
        need = {}
        for v, t in self.rows:
            need.setdefault(v, set()).update([t, t + 1])
        self.need = {v: sorted(ts) for v, ts in need.items()}

    def compute(self, model, device, autocast):
        feats, row = frame_features(model, self.videos, self.need, device, autocast)
        now, nxt = [row[(v, t)] for v, t in self.rows], [row[(v, t + 1)] for v, t in self.rows]
        vid = np.array([v for v, _ in self.rows])
        out = {}
        for name, f in feats.items():
            x = pca(f, torch.ones(len(f), dtype=torch.bool), k=64)
            inputs = {"frame": x[now], "pair": torch.cat([x[now], x[nxt] - x[now]], 1)}
            for target, y in self.targets.items():
                pred = heldout(inputs["pair" if target.startswith("d") else "frame"], y, vid)
                out[f"{target}_r2_{name}"] = r2(pred, y)
        return out
