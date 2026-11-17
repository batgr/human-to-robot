import hashlib
import io
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset


def decode(jpegs):
    """JPEG bytes -> (T, 3, H, W) uint8 frames"""
    frames = np.stack([np.asarray(Image.open(io.BytesIO(j))) for j in jpegs])
    return torch.from_numpy(frames).permute(0, 3, 1, 2)


def shards(root, split):
    return sorted(Path(root, "frames").glob(f"{split}-*.parquet"))


def download(repo, root):
    """Frame shards of a Hub dataset into root, once (the raw videos and simulator states stay on the Hub)"""
    if not shards(root, "train"):
        from huggingface_hub import snapshot_download

        snapshot_download(repo, repo_type="dataset", local_dir=root,
                          allow_patterns=["frames/*", "manifest.parquet", "config.yaml"])


def allocate(available, hours):
    """Hours taken from each subset: equal shares, a subset with less gives all it has and the others share the rest"""
    take, left = {}, hours
    for i, (s, a) in enumerate(sorted(available.items(), key=lambda x: x[1])):
        take[s] = min(a, left / (len(available) - i))
        left -= take[s]
    return take


def select(manifest, share, seed, subset=""):
    """Rows of a subset's manifest making up share (0-1) of each split: whole shards in a seeded order, the videos
    of the last in a seeded order (extract shuffles the videos, a shard is a random sample). The same seed with a
    larger share keeps the smaller selection"""
    rank = lambda key: hashlib.md5(f"{seed}:{subset}/{key}".encode()).hexdigest()
    out = []
    for split in sorted({r["split"] for r in manifest}):
        rows = sorted((r for r in manifest if r["split"] == split), key=lambda r: (rank(r["shard"]), rank(r["video"])))
        target, got = share * sum(r["n_frames"] for r in rows), 0
        for r in rows:
            if got >= target:
                break
            out.append(r)
            got += r["n_frames"]
    return out


def fetch(repo, root, subsets=None, hours=None, seed=0):
    """Part of a Hub dataset of several subsets (<subset>/frames/, manifest.parquet, config.yaml, pushed by
    h2r data push with subset) into root, in ClipDataset's layout (frames/<split>-<subset>-NNNNN.parquet).
    subsets: a name, a list, or None = all of them. hours: total over the splits, split equally among the subsets
    (allocate), each split taking the same share of a subset; None = everything. Only the shards needed are
    downloaded. root/manifest.parquet gains a subset column; the subsets' other parquet files (labels, hands) are
    filtered to the kept videos and concatenated. Once per root: fetch.yaml records the request"""
    import pyarrow.compute as pc
    import yaml
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    root = Path(root)
    request = {"repo": repo, "subsets": subsets, "hours": hours, "seed": seed}
    if (root / "fetch.yaml").exists():
        if yaml.safe_load((root / "fetch.yaml").read_text())["request"] != request:
            raise ValueError(f"{root} holds another selection (fetch.yaml): take another root")
        return
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    every = sorted(f.split("/")[0] for f in files if f.count("/") == 1 and f.endswith("/manifest.parquet"))
    names = every if subsets is None else [subsets] if isinstance(subsets, str) else list(subsets)
    if set(names) - set(every):
        raise ValueError(f"no subset {sorted(set(names) - set(every))} in {repo}, which has {every}")

    get = lambda f: hf_hub_download(repo, f, repo_type="dataset")
    frames = {}  # how the frames were made: subsets mixed only when they agree
    for s in names:
        c = yaml.safe_load(Path(get(f"{s}/config.yaml")).read_text())
        frames[s] = {"fps": c["fps"], "image_size": c["image_size"], "crop": c.get("crop", True)}
    if any(f != frames[names[0]] for f in frames.values()):
        raise ValueError(f"subsets with different frames: {frames}")
    fps = frames[names[0]]["fps"]
    manifests = {s: pq.read_table(get(f"{s}/manifest.parquet")).to_pylist() for s in names}
    available = {s: sum(r["n_frames"] for r in m) / fps / 3600 for s, m in manifests.items()}
    take = available if hours is None else allocate(available, hours)
    kept = {s: select(manifests[s], take[s] / available[s], seed, s) for s in names}

    extras = [f for f in files if f.count("/") == 1 and f.split("/")[0] in names
              and f.endswith(".parquet") and not f.endswith("/manifest.parquet")]
    tmp = root / ".download"
    snapshot_download(repo, repo_type="dataset", local_dir=tmp, allow_patterns=extras + sorted(
        {f"{s}/frames/{r['shard']}" for s in names for r in kept[s]}))

    (root / "frames").mkdir(parents=True, exist_ok=True)
    manifest = []
    for s in names:
        for shard in sorted({r["shard"] for r in kept[s]}):
            videos = [r["video"] for r in kept[s] if r["shard"] == shard]
            name = shard.replace("-", f"-{s}-", 1)  # train-00003 -> train-<subset>-00003
            path = tmp / s / "frames" / shard
            if len(videos) == sum(r["shard"] == shard for r in manifests[s]):
                path.replace(root / "frames" / name)
            else:  # the last shard of a split, trimmed
                table = pq.read_table(path)
                pq.write_table(table.filter(pc.is_in(table["video"], pa.array(videos))), root / "frames" / name)
            manifest += [{"subset": s, "video": r["video"], "split": r["split"], "n_frames": r["n_frames"],
                          "shard": name} for r in kept[s] if r["shard"] == shard]
    pq.write_table(pa.Table.from_pylist(manifest), root / "manifest.parquet")
    for extra in sorted({f.split("/")[1] for f in extras}):
        tables = []
        for s in names:
            if (tmp / s / extra).exists():
                t = pq.read_table(tmp / s / extra)
                videos = pa.array([r["video"] for r in kept[s]])
                tables.append(t.filter(pc.is_in(t["video"], videos)) if "video" in t.column_names else t)
        pq.write_table(pa.concat_tables(tables, promote_options="default"), root / extra)
    shutil.rmtree(tmp)

    got = {s: {split: round(sum(r["n_frames"] for r in kept[s] if r["split"] == split) / fps / 3600, 3)
               for split in ("train", "val", "test")} for s in names}
    (root / "fetch.yaml").write_text(yaml.safe_dump({"request": request, "frames": frames[names[0]], "hours": got}))
    for s in names:
        print(f"{s}: {take[s]:.2f} h of {available[s]:.2f} h ({got[s]})")


def read_episodes(shard):
    """Videos of a shard that have actions: [{"video", "frames": JPEG bytes, "actions": (T - 1, act_dim)}], and
    "states": (T, state_dim) the absolute state at each frame when the shard has it"""
    columns = [c for c in ["video", "frames", "actions", "states"] if c in pq.read_schema(shard).names]
    return [r for r in pq.read_table(shard, columns=columns).to_pylist() if r["actions"] is not None]


def windows(episodes, horizon):
    """Non-overlapping windows of horizon + 1 states of every episode, as many as fit. episodes: {"s": states of each
    episode, (T_e, ...), the others per frame (T_e, ...) e.g. "x" or per transition (T_e - 1, ...) e.g. "a", "mu"}:
    lists, the episodes may differ in length (DROID), or tensors (E, T, ...) -> per frame (W, horizon + 1, ...), per
    transition (W, horizon, ...).
    Ordered by start time then episode: neighbouring windows come from different episodes (ControlledRollout takes
    another episode's actions from the neighbour)"""
    out = {k: [] for k in episodes}
    for t in range(0, max(len(s) for s in episodes["s"]) - horizon, horizon):
        for e, s in enumerate(episodes["s"]):
            if t + horizon < len(s):
                for k, v in episodes.items():
                    out[k].append(v[e][t : t + horizon + (len(v[e]) == len(s))])
    return {k: torch.stack(v) for k, v in out.items()}


def arrow(shard):
    """Uncompressed Arrow IPC copy of a shard's frames next to it (.arrow), written once: unlike parquet it can be
    memory-mapped without decoding. large_binary: 64-bit offsets, no 2 GB limit"""
    path = Path(shard).with_suffix(".arrow")
    if not path.exists():
        frames = pq.read_table(shard, columns=["frames"]).column("frames")
        table = pa.table({"frames": frames.cast(pa.list_(pa.large_binary()))})
        tmp = path.with_suffix(".tmp")
        with pa.OSFile(str(tmp), "wb") as f, pa.ipc.new_file(f, table.schema) as writer:
            writer.write_table(table)
        tmp.replace(path)  # a crash never leaves a half-written .arrow
    return path


class ClipDataset(Dataset):
    """Clips of clip_len consecutive frames of every video, one starting every stride frames (stride 1: every start,
    neighbours share clip_len - 1 frames). The frames are memory-mapped from the .arrow copy of each shard: the RAM
    does not grow with the dataset (the OS pages in what is read), and a DataLoader worker reopens the files instead
    of receiving a pickled copy"""

    def __init__(self, root, split, clip_len=5, stride=1):
        self.paths = [arrow(p) for p in shards(root, split)]
        self.clip_len, self.stride = clip_len, stride
        self.chunks = self.open()
        lengths = np.concatenate([c.value_lengths().to_numpy() for c in self.chunks])
        self.where = np.concatenate(  # video -> (chunk, row)
            [np.stack([np.full(len(c), k), np.arange(len(c))], 1) for k, c in enumerate(self.chunks)])
        self.ends = np.cumsum(np.where(lengths >= clip_len, (lengths - clip_len) // stride + 1, 0))

    def open(self):
        return [c for p in self.paths for c in pa.ipc.open_file(pa.memory_map(str(p))).read_all().column("frames").chunks]

    def __getstate__(self):
        return self.__dict__ | {"chunks": None}  # workers reopen the memory maps

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, i):
        """Returns (clip_len, 3, H, W) uint8"""
        if self.chunks is None:
            self.chunks = self.open()
        v = int(np.searchsorted(self.ends, i, side="right"))
        s = (i - (int(self.ends[v - 1]) if v else 0)) * self.stride
        k, r = self.where[v]
        return decode(j.as_py() for j in self.chunks[k][int(r)].values[s : s + self.clip_len])
