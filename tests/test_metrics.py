import math

import torch
from test_jepa import small_model


def test_perfect_prediction_has_the_true_change():
    """pred = s_t+1: the predicted change is the true one (size 1, direction 1), the next frame is retrieved, no error"""
    from h2r.metrics.prediction import PredictionQuality

    torch.manual_seed(0)
    emb = torch.randn(4, 6, 16)
    clip = torch.randint(0, 255, (4, 6, 3, 8, 8), dtype=torch.uint8)
    q = PredictionQuality()
    q.update(emb, emb[:, 1:], clip)
    out = q.compute()
    for name in ["", "_static", "_middle", "_moving"]:
        assert math.isclose(out[f"change_ratio{name}"], 1, rel_tol=1e-5)
        assert math.isclose(out[f"change_cos{name}"], 1, rel_tol=1e-5)
        assert out[f"retrieval{name}"] == 1
        assert out[f"tf_over_drift{name}"] == 0


def test_persistence_invents_no_motion():
    """pred = s_t: no predicted change at all"""
    from h2r.metrics.prediction import PredictionQuality

    torch.manual_seed(0)
    emb = torch.randn(4, 6, 16)
    q = PredictionQuality()
    q.update(emb, emb[:, :-1], torch.randint(0, 255, (4, 6, 3, 8, 8), dtype=torch.uint8))
    out = q.compute()
    assert out["invented_static"] == 0 and out["change_ratio_static"] == 0
    assert math.isclose(out["tf_over_persistence"], 1, rel_tol=1e-5)


def test_idm_forward_is_its_posterior():
    """eval: z = mu; the KL is the one of (mu, logvar)"""
    model, D = small_model()
    emb = torch.randn(2, 5, D)
    z, kl = model.idm(emb)
    mu, logvar = model.idm.posterior(emb)
    assert torch.equal(z, mu)
    assert torch.allclose(kl, 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(-1).mean())


def test_latent_action_and_predictor_metrics_are_finite():
    from h2r.metrics.latent_actions import LatentActions
    from h2r.metrics.prediction import PredictorInternals

    model, D = small_model()
    emb = torch.randn(6, 5, D)
    z, kl = model.idm(emb)
    latent, internals = LatentActions(), PredictorInternals()
    with torch.no_grad():
        latent.update(model, emb, z, model.predict(emb[:, :-1], z), kl)
        internals.update(model, emb, z)
    out = latent.compute() | internals.compute()
    assert {"leak", "transfer_cos", "mu_spread_over_sigma", "gate_msa", "gate_mlp", "blocks_share"} <= set(out)
    assert all(math.isfinite(v) for v in out.values())
    assert 0 <= out["leak"] <= 1


def test_motion_probe_on_a_tiny_split(tmp_path):
    """MotionProbe reads whole validation videos and labels: every metric is there and finite, the class chance is
    1 / the number of classes"""
    import io

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    from h2r.metrics.probe import MotionProbe

    rng = np.random.default_rng(0)

    def jpeg():
        buf = io.BytesIO()
        Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).save(buf, format="JPEG")
        return buf.getvalue()

    names = ["Moving something up", "Moving something down", "Pushing something from left to right"]
    rows = [{"video": f"{i}.webm", "frames": [jpeg() for _ in range(6)]} for i in range(60)]
    (tmp_path / "frames").mkdir()
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "frames" / "val-00000.parquet")
    pq.write_table(pa.Table.from_pylist([{"video": r["video"], "label": names[i % 3]} for i, r in enumerate(rows)]),
                   tmp_path / "labels.parquet")
    def walk():  # small steps: no tracker jump
        return (0.5 + 0.03 * rng.standard_normal(6).cumsum()).tolist()

    hands = [{"video": r["video"], "x": walk(), "y": walk()} for r in rows]
    pq.write_table(pa.Table.from_pylist(hands), tmp_path / "hands.parquet")

    model, D = small_model()
    proj = torch.randn(3, D)
    model.encode = lambda px: px.float().mean((-1, -2)) @ proj  # (B, T, 3) -> (B, T, D)
    out = MotionProbe(tmp_path, videos=50, hands=tmp_path / "hands.parquet").compute(model, torch.device("cpu"),
                                                                                    torch.autocast("cpu", enabled=False))
    assert {"z_from_change_r2", "motion_r2_z", "motion_r2_state", "reverse_cos", "within_clip_share", "class_z",
            "class_state", "up_down_z", "hand_motion_r2_z"} <= set(out)
    assert math.isclose(out["class_chance"], 1 / 3)
    assert all(math.isfinite(v) for v in out.values())


def test_classifiers_separate_clean_classes():
    from h2r.metrics.probe import classify

    torch.manual_seed(0)
    y = torch.randint(6, (600,))
    x = torch.nn.functional.one_hot(y, 6).float() * 3 + 0.1 * torch.randn(600, 6)
    out = classify(x[:400], y[:400], x[400:], y[400:], 6)
    assert out["kNN"][0] > 0.99 and out["linear"][0] > 0.99
