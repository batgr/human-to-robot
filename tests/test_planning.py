import torch
from torch import nn

from h2r.models.controller import Controller
from h2r.models.idm import IDM
from h2r.models.jepa import JEPA
from h2r.models.module import ARPredictor
from h2r.plan import context_latents, delta_xyz, plan

N, D, A = 16, 32, 7


def setup():
    torch.manual_seed(0)
    predictor = ARPredictor(num_patches=N, depth=2, heads=2, mlp_dim=2 * D, input_dim=D, hidden_dim=D, dim_head=16)
    for block in predictor.transformer.layers:  # non-zero AdaLN, as after training
        nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
    model = JEPA(encoder=None, predictor=predictor, idm=IDM(num_patches=N, dim=D)).eval()
    ctrl = Controller(num_patches=N, dim=D, act_dim=A).eval()
    nn.init.normal_(ctrl.head.weight, std=0.1)
    return model, ctrl


def test_rollout_matches_a_step_by_step_loop():
    """Batched rollout == one candidate at a time: window of 3, controller on the state each action starts from"""
    model, ctrl = setup()
    emb, actions = torch.randn(2, 3, N, D), torch.randn(2, 4, 5, A)  # B = 2, S = 4 candidates, H = 5
    latents = ctrl(torch.randn(2, 2, A), emb[:, :-1])
    out = model.rollout(emb, latents, actions, ctrl, history_size=3)
    assert out.shape == (2, 4, 5, N, D)
    for b in range(2):
        for s in range(4):
            states, z = list(emb[b]), list(latents[b])
            for i in range(5):
                z.append(ctrl(actions[b, s, i][None], states[-1][None])[0])
                states.append(model.predict(torch.stack(states[-3:])[None], torch.stack(z[-3:])[None])[0, -1])
                assert torch.allclose(out[b, s, i], states[-1], atol=1e-5)


def test_context_latents():
    model, ctrl = setup()
    emb, past = torch.randn(3, N, D), torch.randn(2, A)
    assert torch.allclose(context_latents(model, ctrl, emb, past, "controller"), ctrl(past, emb[:-1]))
    assert torch.allclose(context_latents(model, ctrl, emb, past, "idm"), model.idm(emb[None])[0][0])
    one = context_latents(model, ctrl, emb[:1], past[:0], "controller")  # a single context frame
    assert one.shape == (0, D)
    assert model.rollout(emb[None, :1], one[None], torch.randn(1, 3, 2, A), ctrl).shape == (1, 3, 2, N, D)


def test_criterion():
    model, _ = setup()
    pred, goal = torch.zeros(1, 2, 3, N, D), torch.ones(1, N, D)
    pred[0, 1, -1] = 1.0
    assert model.criterion(pred, goal).tolist() == [[N * D, 0.0]]


class Quadratic:
    """Stands in for the world model: the cost of a sequence is its distance to a target sequence"""

    def __init__(self, target):
        self.target = target

    def get_cost(self, emb, latents, goal, actions, ctrl, history_size):
        return (actions - self.target).pow(2).sum(dim=(-2, -1))


CFG = {"horizon": 3, "samples": 300, "elites": 10, "iterations": 15, "searched": [0, 1, 2, 6], "max_norm": None,
       "pick": "best", "window": 3}


def test_cem_in_physical_units():
    """DROID-like scale (centimeters): sampling in standardized units still converges; rotations stay at 0"""
    ctrl = Controller(num_patches=N, dim=D, act_dim=A)
    ctrl.act_std.fill_(0.01)
    target = torch.tensor([0.01, -0.02, 0.015, 0.3, 0.3, 0.3, 0.005]).expand(3, A)
    actions = plan(Quadratic(target), ctrl, *torch.zeros(3, 1), CFG, torch.Generator().manual_seed(0))
    assert actions.shape == (3, A) and torch.equal(actions[:, 3:6], torch.zeros(3, 3))
    assert torch.allclose(actions[:, [0, 1, 2, 6]], target[:, [0, 1, 2, 6]], atol=2e-3)


def test_cem_norm_bound():
    ctrl = Controller(num_patches=N, dim=D, act_dim=A)
    actions = plan(Quadratic(torch.full((2, A), 5.0)), ctrl, *torch.zeros(3, 1), CFG | {"horizon": 2, "max_norm": 0.1})
    assert (actions[:, :3].norm(dim=-1) <= 0.1 + 1e-6).all()


def test_delta_xyz():
    real = [[0.01, 0.0, 0.0, 0, 0, 0, 0]] * 3
    assert abs(delta_xyz([[0.0] * 7] * 3, real) - 0.03) < 1e-9 and delta_xyz(real, real) == 0.0
