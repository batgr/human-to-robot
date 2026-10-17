"""robosuite environment: the collection (h2r data collect) goes through this class"""

import numpy as np
import robosuite as suite
from scipy.spatial.transform import Rotation


class RobosuiteEnv:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.env = suite.make(**self.cfg)

    def reset(self):
        return self.with_images(self.env.reset())

    def step(self, action, repeat=1):
        """Holds the action for repeat control steps (frameskip); returns the last step"""
        for _ in range(repeat):
            obs, reward, done, info = self.env.step(action)
            if self.cfg.get("has_renderer", False):
                self.env.render()
        return self.with_images(obs), reward, done, info

    def with_images(self, obs):
        """With use_camera_obs: false, the camera images are rendered here, once per returned observation, instead of
        by robosuite at every control step (5 renders per frame kept at frameskip 5)"""
        if self.cfg.get("has_offscreen_renderer") and not self.cfg.get("use_camera_obs", True):
            for cam in self.cfg["camera_names"]:
                obs[f"{cam}_image"] = self.env.sim.render(
                    camera_name=cam, width=self.cfg["camera_widths"], height=self.cfg["camera_heights"])
        return obs

    @staticmethod
    def image(obs, camera):
        return obs[f"{camera}_image"][::-1].copy()  # robosuite renders upside down

    def gripper(self):
        return next(iter(self.env.robots[0].gripper.values()))

    def sim_state(self):
        """MuJoCo state, then the gripper's command integrator (robosuite keeps it outside the MuJoCo state)"""
        return np.r_[self.env.sim.get_state().flatten(), self.gripper().current_action]

    def robot_state(self, obs):
        """[x, y, z, roll, pitch, yaw, gripper] in the robot base frame, as DROID's cartesian and gripper positions"""
        sim, robot = self.env.sim, self.env.robots[0]
        base = sim.model.body_name2id(robot.robot_model.root_body)
        base_pos, base_rot = sim.data.body_xpos[base], Rotation.from_matrix(
            sim.data.body_xmat[base].reshape(3, 3)
        )
        pos = base_rot.inv().apply(obs["robot0_eef_pos"] - base_pos)
        rot = (base_rot.inv() * Rotation.from_quat(obs["robot0_eef_quat"])).as_euler(
            "xyz"
        )  # quat: x, y, z, w
        joints = next(iter(robot.gripper.values())).joints
        ranges = [sim.model.jnt_range[sim.model.joint_name2id(j)] for j in joints]
        q = obs["robot0_gripper_qpos"]
        closed = 1 - (q[0] - q[1]) / (
            ranges[0][1] - ranges[1][0]
        )  # 0: fingers fully apart, 1: touching
        return np.r_[pos, rot, np.clip(closed, 0, 1)]


if __name__ == "__main__":
    from pathlib import Path

    import yaml

    cfg_path = Path(__file__).parents[3] / "configs" / "env" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    env = RobosuiteEnv(cfg)
    env.reset()
    for _ in range(1000):
        env.step(np.random.randn(7))
