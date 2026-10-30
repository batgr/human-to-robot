import pytest
import torch
import yaml
from torch import nn

from h2r.models.controller import Controller
from h2r.models.module import ARPredictor

D, A = 32, 7


@pytest.fixture
def ctrl():
    torch.manual_seed(0)
    return Controller(dim=D, act_dim=A)


def trained(ctrl):
    """Non-zero head, as after training"""
    nn.init.normal_(ctrl.head.weight, std=0.1)
    return ctrl


def test_shapes(ctrl):
    assert ctrl(torch.randn(4, A), torch.randn(4, D)).shape == (4, D)
    assert ctrl(torch.randn(4, 3, A), torch.randn(4, 3, D)).shape == (4, 3, D)


def test_starts_at_zero(ctrl):
    assert torch.equal(ctrl(torch.randn(5, A), torch.randn(5, D)), torch.zeros(5, D))


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
    a, s = torch.randn(1, A), torch.randn(1, D)
    z = ctrl(a, s)
    assert not torch.allclose(z, ctrl(a, torch.randn(1, D)))
    assert not torch.allclose(z, ctrl(torch.randn(1, A), s))


def test_deterministic(ctrl):
    ctrl = trained(ctrl).train()
    a, s = torch.randn(3, A), torch.randn(3, D)
    assert torch.equal(ctrl(a, s), ctrl(a, s))


def test_drives_the_predictor(ctrl):
    """z_hat replaces the IDM's z in the predictor"""
    ctrl = trained(ctrl)
    predictor = ARPredictor(depth=1, heads=2, mlp_dim=2 * D, input_dim=D, hidden_dim=D, dim_head=16)
    emb = torch.randn(2, 3, D)
    assert predictor(emb, ctrl(torch.randn(2, 3, A), emb)).shape == (2, 3, D)


def test_default_config():
    cfg = yaml.safe_load(open("configs/models/controller.yaml"))
    assert Controller(dim=D, **cfg)(torch.randn(2, cfg["act_dim"]), torch.randn(2, D)).shape == (2, D)
