import torch
import torch.nn.functional as F
from test_jepa import small_model


def test_rollout_terms_match_an_explicit_unrolling():
    """L_k = error on state j after k steps from the real state j - k, each step in a window of w states
    that holds the previous predictions; every j >= k is a target"""
    from h2r.training.world_model import rollout_terms

    model, D = small_model()
    T, w = 7, 3
    emb, z = torch.randn(2, T, D), torch.randn(2, T - 1, D)

    def unroll(j, k):
        """prediction of state j after k steps from j - k, windows clipped at the start of the clip"""
        states = {p: emb[:, p] for p in range(j - k + 1)}
        for q in range(j - k + 1, j + 1):  # predict state q from the window before it
            a = max(0, q - w)
            ctx = torch.stack([states[p] for p in range(a, q)], dim=1)
            states[q] = model.predict(ctx, z[:, a:q])[:, -1]
        return states[j]

    terms = rollout_terms(model, emb, z, steps=4, w=w)
    assert len(terms) == 3  # L_2, L_3, L_4
    for k, term in enumerate(terms, start=2):
        expected = torch.stack([unroll(j, k) for j in range(k, T)], dim=1)
        assert torch.allclose(term, F.mse_loss(expected, emb[:, k:]), atol=1e-5)
    assert rollout_terms(model, emb, z, steps=1, w=w) == []


def test_stop_grad_target_cuts_only_the_target_path():
    """loss.stop_grad_target: same loss value; no gradient of tf reaches a state used only as a target (the IDM's z
    held fixed here), while the states read by the predictor still get one"""
    from h2r.models.module import SIGReg
    from h2r.training.world_model import losses

    model, D = small_model()
    z = torch.randn(2, 3, D)
    model.idm.forward = lambda emb: (z, torch.zeros(()))  # z without a path to the states
    cfg = {"loss": {"beta": 0.0, "sigreg": 0.0, "rollout_steps": 1, "rollout_window": 3}}
    emb = torch.randn(2, 4, D, requires_grad=True)
    grads = {}
    for sg in (False, True):
        out = losses(model, SIGReg(), emb, {"loss": cfg["loss"] | {"stop_grad_target": sg}})
        grads[sg] = (out["tf"].item(), torch.autograd.grad(out["tf"], emb)[0])
    assert grads[False][0] == grads[True][0]
    assert grads[False][1][:, -1].abs().sum() > 0 and grads[True][1][:, -1].abs().sum() == 0  # last state: target only
    assert grads[True][1][:, 0].abs().sum() > 0  # first state: read by the predictor
