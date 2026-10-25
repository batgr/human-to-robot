import torch


def test_rollout_without_controller_takes_latent_actions():
    """JEPA.rollout(controller=None) unrolls the given latent actions, as a loop over predict"""
    from h2r.models.idm import IDM
    from h2r.models.jepa import JEPA
    from h2r.models.module import ARPredictor

    torch.manual_seed(0)
    N, D = 4, 16
    model = JEPA(None, ARPredictor(num_patches=N, depth=1, heads=2, mlp_dim=32, input_dim=D, hidden_dim=D, dim_head=8),
                 IDM(num_patches=N, dim=D)).eval()
    for block in model.predictor.transformer.layers:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
    emb, z = torch.randn(2, 3, N, D), torch.randn(2, 5, D)
    out = model.rollout(emb, z[:, :2], z[:, 2:][:, None], history_size=3)[:, 0]
    ctx = emb
    for h in range(3):
        nxt = model.predict(ctx[:, -3:], z[:, h : h + 3])[:, -1:]
        ctx = torch.cat([ctx, nxt], 1)
    assert torch.allclose(out, ctx[:, 3:], atol=1e-5)
