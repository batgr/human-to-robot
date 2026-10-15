import pytest
import torch
import yaml
from torch import nn

from h2r.models.controller import Controller
from h2r.models.module import ARPredictor

N, D, A = 16, 32, 7


@pytest.fixture
def ctrl():
    torch.manual_seed(0)
    return Controller(num_patches=N, dim=D, act_dim=A)


def trained(ctrl):
    """Non-zero head, as after training"""
    nn.init.normal_(ctrl.head.weight, std=0.1)
    return ctrl


def test_shapes(ctrl):
    assert ctrl(torch.randn(4, A), torch.randn(4, N, D)).shape == (4, D)
    assert ctrl(torch.randn(4, 3, A), torch.randn(4, 3, N, D)).shape == (4, 3, D)
    assert Controller(num_patches=N, dim=D, latent_dim=8)(torch.randn(2, A), torch.randn(2, N, D)).shape == (2, 8)


def test_starts_at_zero(ctrl):
    assert torch.equal(ctrl(torch.randn(5, A), torch.randn(5, N, D)), torch.zeros(5, D))


def test_action_stats(ctrl):
    actions = torch.randn(1000, A) * 0.01 + 0.5
    actions[:, 3:6] = 0  # constant dims, like the rotations in robosuite
    ctrl.set_action_stats(actions)
    assert torch.allclose(ctrl.act_mean, actions.mean(0))
    assert torch.equal(ctrl.act_std[3:6], torch.ones(3))
    state = ctrl.state_dict()
    assert torch.equal(state["act_mean"], ctrl.act_mean) and torch.equal(state["act_std"], ctrl.act_std)


def test_depends_on_state_and_action(ctrl):
    ctrl = trained(ctrl)
    a, s = torch.randn(1, A), torch.randn(1, N, D)
    z = ctrl(a, s)
    assert not torch.allclose(z, ctrl(a, torch.randn(1, N, D)))
    assert not torch.allclose(z, ctrl(torch.randn(1, A), s))
    assert not torch.allclose(z, ctrl(a, s.roll(1, dims=1)))  # where the patches are matters


def test_deterministic(ctrl):
    ctrl = trained(ctrl).train()
    a, s = torch.randn(3, A), torch.randn(3, N, D)
    assert torch.equal(ctrl(a, s), ctrl(a, s))


def test_drives_the_predictor(ctrl):
    """z_hat replaces the IDM's z in the predictor"""
    ctrl = trained(ctrl)
    predictor = ARPredictor(num_patches=N, depth=1, heads=2, mlp_dim=2 * D, input_dim=D, hidden_dim=D, dim_head=16)
    emb = torch.randn(2, 3, N, D)
    assert predictor(emb, ctrl(torch.randn(2, 3, A), emb)).shape == (2, 3, N, D)


def test_default_config():
    cfg = yaml.safe_load(open("configs/models/controller.yaml"))
    assert Controller(num_patches=N, dim=D, **cfg)(torch.randn(2, cfg["act_dim"]), torch.randn(2, N, D)).shape == (2, D)
