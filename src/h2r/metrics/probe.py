"""Linear probes of the hand during training: does the state (and the encoder) keep where the hand is, and the direction
of the clip? The protocol of where.ipynb: SSv2 validation frames with a tracked hand (MediaPipe), tokens pooled on a
4 x 4 grid, a shared PCA of 24 dims per region, ridge held out by video (2 folds by video parity, lambda chosen on an
inner split by video)"""

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



def balanced_accuracy(pred, y, n):
    """Mean recall over the classes present, and the confusion matrix (rows: true class, normalized)"""
    conf = torch.zeros(n, n).index_put_((y, pred), torch.ones(len(y)), accumulate=True)
    present = conf.sum(1) > 0
    recall = conf.diagonal() / conf.sum(1).clamp_min(1)
    return recall[present].mean().item(), conf / conf.sum(1, keepdim=True).clamp_min(1)


def classify(xtr, ytr, xte, yte, n, k=10):
    """Balanced accuracy and confusion of a k-nearest-neighbours vote (Euclidean) and of a linear classifier (ridge on
    one-hot targets), features standardized on the training points: {"kNN": (acc, conf), "linear": (acc, conf)}"""
    m, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = ((x - m) / sd for x in (xtr, xte))
    near = torch.cat([torch.cdist(xte[i : i + 1024], xtr).topk(min(k, len(xtr)), largest=False).indices
                      for i in range(0, len(xte), 1024)])
    knn = torch.mode(ytr[near], 1).values
    x1 = torch.cat([xtr, torch.ones(len(xtr), 1)], 1).double()
    w = torch.linalg.solve(x1.T @ x1 + 1e-2 * len(x1) * torch.eye(x1.shape[1], dtype=torch.double),
                           x1.T @ F.one_hot(ytr, n).double())
    linear = (torch.cat([xte, torch.ones(len(xte), 1)], 1).double() @ w).argmax(1)
    return {"kNN": balanced_accuracy(knn, yte, n), "linear": balanced_accuracy(linear, yte, n)}


def fold_accuracy(x, y, vid, n):
    """Balanced accuracy of a linear classifier (ridge on one-hot targets) held out by video: 2 folds by video
    parity, the mean of the two"""
    accs = []
    for fold in (0, 1):
        fit = torch.as_tensor(vid % 2 == fold)
        if fit.all() or not fit.any():
            return float("nan")
        accs.append(classify(x[fit], y[fit], x[~fit], y[~fit], n)["linear"][0])
    return float(np.mean(accs))


class MotionProbe:
    """What the latent action z and the states know of the motion, on up to `videos` whole validation videos (every
    transition of each), linear probes held out by video (2 folds by video parity). Features: z = the IDM's mu,
    change = Δs = s_t+1 - s_t, state = s_t (states pooled 4 x 4, a PCA of 64 dims shared by the regions).
    Without labels:
    - z_from_change_r2: R² of mu from Δs, z as a function of the change of state (diagnostics 3 Nov.: CLS 0.86,
      patch16 0.82, patch16 + stop-grad 0.00, with a PCA of 128 dims there);
    - motion_r2_{z, change, state}: R² of the log pixel motion (mean |frame t+1 - frame t|): how much moves; from
      |Δs| for change (a linear map of the signed Δs cannot read a size). state is the control: what the scene alone
      tells of it;
    - reverse_cos: median cosine between mu(s_t, s_t+1) and mu(s_t+1, s_t), both centred by the mean mu: -1 = reversing
      time reverses z, as for an action; +1 = z ignores the order (a code of the pair, of the scene);
    - within_clip_share: share of mu's variance within the videos; 0 = one z per video (a code of the clip, not of
      each transition).
    With labels.parquet (SSv2 classes):
    - class_{z, change, state}: balanced accuracy of the class from the video's mean (chance class_chance). z above
      state = z names the action beyond the scene; change near chance = the encoder does not see the motion;
    - left_right_{z, change}, up_down_{z, change}: accuracy of the direction (HandProbe's classes, chance 0.5).
    With hands (parquet of video, x, y per frame):
    - hand_motion_r2_{z, change}: R² of the tracked hand's displacement (dx, dy in the image) on the transitions with
      a hand in both frames and a step under 0.25 of the image (where.ipynb; 10h-20cls: median 0.055, 13 % above 0.2, the
      tracker jumping between hands or onto noise).
    Returns {} while mu is still 0 (the IDM's head starts at zero)."""

    def __init__(self, root, videos=1000, hands=None, seed=0):
        rows = [r for p in shards(root, "val") for r in pq.read_table(p, columns=["video", "frames"]).to_pylist()
                if len(r["frames"]) >= 3]
        pick = np.random.default_rng(seed).choice(len(rows), min(videos, len(rows)), replace=False)
        self.videos = [rows[i] for i in sorted(pick)]
        self.motion = []  # mean absolute pixel change of each transition
        for r in self.videos:
            px = decode(r["frames"]).float()
            self.motion.append((px[1:] - px[:-1]).abs().mean((1, 2, 3)))
        try:
            labels = dict(zip(*[pq.read_table(f"{root}/labels.parquet").column(c).to_pylist() for c in ["video", "label"]]))
        except FileNotFoundError:
            labels = {}
        names = sorted({labels[r["video"]] for r in self.videos if r["video"] in labels})
        self.label = torch.tensor([names.index(labels[r["video"]]) if r["video"] in labels else -1 for r in self.videos])
        self.names = names
        self.directions = {task: torch.tensor([m.get(labels.get(r["video"]), 0) for r in self.videos])
                           for task, m in DIRECTIONS.items()}
        self.hand = None
        if hands:
            track = {r["video"]: np.stack([r["x"], r["y"]], 1) for r in pq.read_table(hands).to_pylist()}
            self.hand = [torch.tensor(np.diff(track[r["video"]], axis=0)) if r["video"] in track
                         else torch.full((len(r["frames"]) - 1, 2), float("nan")) for r in self.videos]

    @torch.no_grad()
    def compute(self, model, device, autocast):
        was_training = model.training
        model.eval()
        s, mu, mu_rev = [], [], []
        for r in self.videos:
            with autocast:
                emb = model.encode(decode(r["frames"]).to(device)[None])
                mu.append(model.idm.posterior(emb)[0][0].float().cpu())
                mu_rev.append(model.idm.posterior(emb.flip(1))[0][0].flip(0).float().cpu())  # (t+1 -> t) at t
            s.append(grid(emb[0]).float().cpu())
        model.train(was_training)
        n_tr = [len(m) for m in mu]
        vid = np.repeat(np.arange(len(self.videos)), n_tr)  # video of each transition
        owner = torch.as_tensor(vid)
        states = torch.cat(s).double()
        frames = pca(states, torch.ones(len(states), dtype=torch.bool), k=64)  # (frames, 64 per region)
        start = np.cumsum([0] + [n + 1 for n in n_tr])
        x = {"z": torch.cat(mu).double(),
             "change": torch.cat([frames[a + 1 : b] - frames[a : b - 1] for a, b in zip(start[:-1], start[1:])]),
             "state": torch.cat([frames[a : b - 1] for a, b in zip(start[:-1], start[1:])])}
        z = x["z"] - x["z"].mean(0)
        if z.pow(2).sum() < 1e-12:  # mu = 0 (the IDM's head at initialization): nothing to probe in z
            return {}
        out = {"z_from_change_r2": r2(heldout(x["change"], x["z"], vid), x["z"])}
        motion = torch.cat(self.motion).double().clamp_min(1e-3).log()[:, None]
        for k, f in [("z", x["z"]), ("change", x["change"].abs()), ("state", x["state"])]:  # a size: |Δs|
            out[f"motion_r2_{k}"] = r2(heldout(f, motion, vid), motion)
        zr = torch.cat(mu_rev).double()
        out["reverse_cos"] = float(F.cosine_similarity(z, zr - zr.mean(0), dim=-1).median())
        means = torch.stack([m.double().mean(0) for m in mu])
        within = sum((m.double() - mm).pow(2).sum() for m, mm in zip(mu, means))
        out["within_clip_share"] = float(within / z.pow(2).sum().clamp_min(1e-12))

        per_video = {k: torch.stack([v[owner == i].mean(0) for i in range(len(self.videos))]) for k, v in x.items()}
        videos = np.arange(len(self.videos))
        known = self.label >= 0
        if known.sum() >= 20 and len(self.names) >= 2:
            out["class_chance"] = 1 / len(self.label[known].unique())
            for k, f in per_video.items():
                out[f"class_{k}"] = fold_accuracy(f[known], self.label[known], videos[known.numpy()], len(self.names))
        for task, d in self.directions.items():
            keep = d != 0
            if keep.sum() < 20:
                continue
            y = d[keep].double()[:, None]
            for k in ("z", "change"):
                pred = heldout(per_video[k][keep], y, videos[keep.numpy()])
                out[f"{task}_{k}"] = float((torch.sign(pred) == y).double().mean())
        if self.hand is not None:
            dxy = torch.cat(self.hand).double()
            seen = ~dxy.isnan().any(1) & (dxy.norm(dim=1) < 0.25)  # tracked in both frames, no jump (where.ipynb MAX_JUMP)
            if seen.sum() >= 50:
                for k in ("z", "change"):
                    out[f"hand_motion_r2_{k}"] = r2(heldout(x[k][seen], dxy[seen], vid[seen.numpy()]), dxy[seen])
        return out
