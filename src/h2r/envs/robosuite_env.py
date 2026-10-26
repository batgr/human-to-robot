"""robosuite environment"""

import numpy as np
import robosuite as suite
from scipy.spatial.transform import Rotation


class RobosuiteEnv:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.env = suite.make(**self.cfg)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs = self.env.step(action)[0]
        if self.cfg.get("has_renderer"):
            self.env.render()
        return obs

    def robot_state(self, obs):

        robot = self.env.robots[0]

        base = self.env.sim.model.body_name2id(robot.robot_model.root_body)
        base_pos = self.env.sim.data.body_xpos[base]
        base_rot = Rotation.from_matrix(self.env.sim.data.body_xmat[base].reshape(3, 3))

        pos = base_rot.inv().apply(obs["robot0_eef_pos"] - base_pos)
        rot = (base_rot.inv() * Rotation.from_quat(obs["robot0_eef_quat"])).as_euler(
            "xyz"
        )

        joints = list(robot.gripper.values())[0].joints
        ranges = [
            self.env.sim.model.jnt_range[self.env.sim.model.joint_name2id(j)]
            for j in joints
        ]
        q = obs["robot0_gripper_qpos"]
        closed = 1 - (q[0] - q[1]) / (ranges[0][1] - ranges[1][0])
        return np.r_[pos, rot, np.clip(closed, 0, 1)]


if __name__ == "__main__":
    from pathlib import Path

    import yaml

    cfg_path = Path(__file__).parents[3] / "configs" / "env" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    env = RobosuiteEnv(cfg)
    obs = env.reset()
    for _ in range(1000):
        action = env.robot_state(obs) + np.random.randn(7)
        action[-1] = np.clip(action[-1], 0, 1)
        obs = env.step(action)
