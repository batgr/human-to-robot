import torch


def small_model():
    """A small JEPA without encoder, the AdaLN and IDM heads moved away from their zero initialization"""
    from h2r.models.idm import IDM
    from h2r.models.jepa import JEPA
    from h2r.models.module import ARPredictor

    torch.manual_seed(0)
    D = 16
    model = JEPA(None, ARPredictor(depth=1, heads=2, mlp_dim=32, input_dim=D, hidden_dim=D, dim_head=8),
                 IDM(dim=D)).eval()
    for block in model.predictor.transformer.layers:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
    torch.nn.init.normal_(model.idm.head.weight, std=0.1)
    return model, D


def test_rollout_without_controller_takes_latent_actions():
    """JEPA.rollout(controller=None) unrolls the given latent actions, as a loop over predict"""
    model, D = small_model()
    emb, z = torch.randn(2, 3, D), torch.randn(2, 5, D)
    out = model.rollout(emb, z[:, :2], z[:, 2:][:, None], history_size=3)[:, 0]
    ctx = emb
    for h in range(3):
        nxt = model.predict(ctx[:, -3:], z[:, h : h + 3])[:, -1:]
        ctx = torch.cat([ctx, nxt], 1)
    assert torch.allclose(out, ctx[:, 3:], atol=1e-5)


def test_predictor_is_causal():
    """Changing frame t (state and latent action) leaves every prediction made before t unchanged"""
    model, D = small_model()
    emb, z = torch.randn(2, 5, D), torch.randn(2, 5, D)
    out = model.predict(emb, z)
    emb2, z2 = emb.clone(), z.clone()
    emb2[:, 3], z2[:, 3] = torch.randn(2, D), torch.randn(2, D)
    out2 = model.predict(emb2, z2)
    assert torch.allclose(out[:, :3], out2[:, :3], atol=1e-6)
    assert not torch.allclose(out[:, 3:], out2[:, 3:])


def test_idm_reads_only_its_pair():
    """z_t depends on frames t and t+1 only"""
    model, D = small_model()
    emb = torch.randn(2, 5, D)
    z, _ = model.idm(emb)
    emb2 = emb.clone()
    emb2[:, 2] = torch.randn(2, D)
    z2, _ = model.idm(emb2)
    assert torch.allclose(z[:, [0, 3]], z2[:, [0, 3]], atol=1e-6)
    assert not torch.allclose(z[:, 1:3], z2[:, 1:3])


def test_idm_on_stacked_pairs_matches_the_clip():
    """The pairs layout of the cycle metric (h2r.metrics.latent_actions) gives the clip's own latent actions"""
    model, D = small_model()
    emb = torch.randn(2, 5, D)
    z, _ = model.idm(emb)
    pairs = torch.stack([emb[:, :-1], emb[:, 1:]], dim=2).flatten(0, 1)  # (B * (T-1), 2, D)
    z_pairs, _ = model.idm(pairs)
    assert torch.allclose(z_pairs.view_as(z), z, atol=1e-6)


def test_latent_actions_smaller_than_the_state():
    """IDM z_dim < D: the predictor reads z through its cond_proj"""
    from h2r.models.idm import IDM
    from h2r.models.module import ARPredictor

    D, Z = 16, 4
    idm = IDM(dim=D, z_dim=Z)
    predictor = ARPredictor(depth=1, heads=2, mlp_dim=32, input_dim=D, hidden_dim=D, dim_head=8, cond_dim=Z)
    emb = torch.randn(2, 5, D)
    z, _ = idm(emb)
    assert z.shape == (2, 4, Z)
    assert predictor(emb[:, :-1], z).shape == (2, 4, D)
