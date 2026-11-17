import numpy as np
import pytest
import torch
from test_jepa import small_model

from h2r.models.controller import Controller
from h2r.plan import (battery, candidate_costs, cem, episode_interval, goal_cost, goal_schedule, other_episodes,
                      plan_step, state_probe, task_windows, to_command, wilson)


def test_cem_finds_the_minimum_of_a_known_cost():
    torch.manual_seed(0)
    target = torch.randn(2, 3, 4)
    cost = lambda plans: ((plans - target[:, None]) ** 2).mean((-1, -2))
    mean, costs, means = cem(cost, torch.zeros(2, 3, 4), torch.ones(2, 3, 4) * 2)
    assert torch.allclose(mean, target, atol=0.1)
    assert costs.shape == (10, 2) and (costs[-1] < costs[0]).all()
    assert means.shape == (10, 2, 3, 4) and torch.equal(means[-1], mean)


def test_goal_cost_is_zero_for_the_plan_that_imagined_the_goal():
    """The goal is the model's own imagined state after a plan: that plan costs 0, another one more"""
    model, D = small_model()
    context, latents, plan = torch.randn(2, 2, D), torch.randn(2, 1, D), torch.randn(2, 3, D)
    goal = model.rollout(context, latents, plan[:, None])[:, 0, -1]
    cost = goal_cost(model, None, context, latents, goal)
    costs = cost(torch.stack([plan, torch.randn(2, 3, D)], dim=1))
    assert torch.allclose(costs[:, 0], torch.zeros(2), atol=1e-6) and (costs[:, 1] > 0).all()


def test_to_command_divides_the_move_and_thresholds_the_gripper():
    K = [0.05, 0.05, 0.1]
    open_ = np.r_[np.zeros(6), -1.0]
    command = to_command([0.01, -0.5, 0.02, 0.3, 0.3, 0.3, 0.2], open_, K, 0.05)
    assert np.allclose(command, [0.2, -1.0, 0.2, 0, 0, 0, 1.0])  # clipped, no rotation, gripper closes
    assert to_command(np.zeros(7), command, K, 0.05)[6] == 1.0  # a small closure change keeps the last command
    assert to_command([0, 0, 0, 0, 0, 0, -0.2], command, K, 0.05)[6] == -1.0


def test_goal_schedule_final_dense_and_stages():
    goals = torch.arange(41.0)[:, None]  # the state of frame t is t
    assert goal_schedule("final", goals, 40)[:, 0].tolist() == [40.0] * 40
    dense = goal_schedule("dense", goals, 40, k=7)[:, 0]
    assert dense[0] == 7 and dense[33] == 40 and dense[-1] == 40
    st = goal_schedule("stages", goals, 40, times={"reach": 10, "grasp": 17, "lift": 18})[:, 0]
    assert (st[:10] == 10).all() and (st[10:17] == 17).all() and (st[17:] == 40).all()


def test_wilson_interval():
    assert wilson(0, 20)[0] == 0 and wilson(20, 20)[1] == 1
    assert np.allclose(wilson(16, 20), [0.584, 0.919], atol=1e-3)


@pytest.mark.parametrize("context", [1, 2, 3])
def test_plan_step_plans_the_horizon_from_any_context(context):
    model, D = small_model()
    model.history_size = context
    controller = Controller(dim=D, z_dim=D).eval()
    torch.nn.init.normal_(controller.head.weight, std=0.1)
    states, actions = [torch.randn(D) for _ in range(2)], [np.zeros(7)]  # two steps in: one achieved action
    plan, cost = plan_step(model, controller, states, actions, torch.randn(D),
                           {"context": context, "horizon": 2, "samples": 20, "elites": 4, "iterations": 2})
    assert plan.shape == (2, 7) and np.isfinite(cost)


def test_collection_policies_replay_the_same_episode():
    """robosuite: the scripted policy with the same seed plays the same episode; the other policies without a model
    run; the run measures every stage"""
    pytest.importorskip("robosuite")
    from h2r.cli import load
    from h2r.envs.robosuite_env import RobosuiteEnv
    from h2r.plan import episode

    cfg = load("configs/planning/robosuite.yaml")
    cfg["steps"], cfg["render"]["video"] = 4, False
    env = RobosuiteEnv(cfg["env"])
    demo = {"video": "test", "seed": 3, "jpegs": None, "actions": np.full((4, 7), 0.01), "commands": [],
            "states": np.zeros((5, 7)), "cube": np.zeros((5, 3))}
    runs = {}
    for kind in ("scripted", "scripted", "random", "recorded", "zero"):
        cfg["policy"]["type"] = kind
        result, _ = episode(env, cfg, demo)
        assert len(result["robot"]) == 5 and set(result["first"]) == {"reach", "grasp", "lift"}
        runs.setdefault(kind, []).append(np.array(result["robot"]))
    assert np.array_equal(runs["scripted"][0], runs["scripted"][1])
    assert np.abs(runs["zero"][0][-1, :3] - runs["zero"][0][1, :3]).max() < 0.01  # the arm stays where it is


def test_episode_interval_draws_episodes():
    episode = torch.tensor([0, 0, 1, 1, 2, 2])
    mean, low, high = episode_interval(torch.tensor([1.0, 1, 0, 0, 1, 1]), episode)
    assert mean == 2 / 3 and low == 0 and high == 1  # 3 episodes: all at 0 or all at 1 can be drawn
    assert episode_interval(torch.full((6,), 0.3), episode) == pytest.approx([0.3, 0.3, 0.3])


def episodes_imagined_by(model, controller, n=6, length=9, D=16):
    """Episodes whose states after the first two are the model's own rollout of their actions"""
    torch.manual_seed(1)
    data = {"s": [], "a": []}
    for _ in range(n):
        s, a = torch.randn(2, D), torch.randn(length - 1, 7)
        z = controller(a[:1], s[:1])
        with torch.no_grad():
            rest = model.rollout(s[None], z[None], a[None, None, 1:], controller)[0, 0]
        data["s"].append(torch.cat([s, rest]))
        data["a"].append(a)
    return data


def test_true_actions_cost_zero_and_rank_first_when_the_episode_is_imagined():
    model, D = small_model()
    model.history_size = 2
    controller = Controller(dim=D, z_dim=64).eval()
    controller.head = torch.nn.Linear(controller.head.in_features, D)  # z of the small model's size
    data = episodes_imagined_by(model, controller, D=D)
    w = task_windows(data, 8, 2, ["scripted", "random"] * 3)
    assert w["s"].shape == (6, 9, D) and set(w["groups"]) == {"all", "calm third", "middle third",
                                                                  "most moving third", "random", "scripted"}
    # the goal is 7 steps after the current state: plan the 7 true actions
    costs, zero = candidate_costs(model, controller, w, other_episodes(w["episode"], 3), 2, 7, "cpu")
    assert torch.allclose(costs[:, 0], torch.zeros(6), atol=1e-6) and (costs[:, 1:] > 0).all() and (zero > 0).all()


def test_state_probe_finds_a_linear_motion():
    torch.manual_seed(0)
    data = {"s": [], "a": []}
    W = torch.randn(4, 16)
    for _ in range(40):
        a = torch.randn(9, 7)
        moved = torch.cat([torch.zeros(1, 4), a[:, [0, 1, 2, 6]].cumsum(0)])
        data["s"].append(moved @ W + torch.randn(16))  # a random start, then the motion
        data["a"].append(a)
    r2 = state_probe(data, data)
    assert min(r2["s_t - s_0"].values()) > 0.99 and max(r2["time only"].values()) < 0.2


def test_battery_runs_on_a_small_model():
    model, D = small_model()
    model.history_size = 2
    controller = Controller(dim=D, z_dim=64).eval()
    controller.head = torch.nn.Linear(controller.head.in_features, D)
    data = episodes_imagined_by(model, controller, n=8, length=17, D=D)
    cfg = {"context": 2, "horizon": 3, "goal": 7, "candidates": 3, "rules": {"near": 0.75, "far": 0.5},
           "cem": {"samples": 20, "elites": 4, "iterations": 2}}
    metrics, saved = battery(model, controller, data, data, None, cfg, "cpu")
    assert metrics["gate_near"]["windows"] == 8 * 4 and metrics["gate_far"]["windows"] == 8 * 2
    assert metrics["gate_near"]["hit_top1"]["all"][0] == 1  # imagined episodes, goal right after: the true actions win
    assert saved["verdict"]["plans"].shape == (16, 3, 7) and isinstance(metrics["verdict"]["passes"], bool)
