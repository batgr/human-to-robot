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
