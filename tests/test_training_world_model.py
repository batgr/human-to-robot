import torch

from h2r.training.world_model import effective_rank, rollout_steps


def test_rollout_curriculum():
    """K goes from start to end one step at a time over the ramp, then stays at end"""
    cfg = {"iterations": 1000, "rollout_steps": {"start": 1, "end": 5, "ramp": 0.5}}
    ks = [rollout_steps(cfg, s) for s in range(1000)]
    assert ks[0] == 1 and ks[499] == 5 and ks[500:] == [5] * 500
    assert [ks.count(k) for k in range(1, 6)] == [100, 100, 100, 100, 600]
    assert rollout_steps(cfg | {"rollout_steps": {"start": 2, "end": 2, "ramp": 0.0}}, 0) == 2


def test_effective_rank():
    """RankMe: about d for isotropic data, about 1 for data on a line"""
    x = torch.randn(4000, 16)
    assert 14 < effective_rank(x) <= 16
    assert effective_rank(torch.randn(4000, 1) * torch.ones(1, 16)) < 1.1


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
