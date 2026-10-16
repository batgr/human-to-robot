"""robosuite environment: the collection (h2r data collect) and the evaluations (h2r eval) go through this class"""

import numpy as np
import robosuite as suite
from scipy.spatial.transform import Rotation


class RobosuiteEnv:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.env = suite.make(**self.cfg)

    def reset(self):
        return self.env.reset()

    def step(self, action, repeat=1):
        """Holds the action for repeat control steps (frameskip); returns the last step"""
        for _ in range(repeat):
            obs, reward, done, info = self.env.step(action)
            if self.cfg.get("has_renderer", False):
                self.env.render()
        return obs, reward, done, info

    @staticmethod
    def image(obs, camera):
        return obs[f"{camera}_image"][::-1].copy()  # robosuite renders upside down

    def sim_state(self):
        return self.env.sim.get_state().flatten()

    def restore(self, sim_state):
        """Puts the simulator back in a recorded state. The controllers cache the robot state and their goal: they are
        refreshed, otherwise the same commands do not replay the same way (1.2 cm apart after one action; refreshed,
        replays stay within a millimeter). Do not call reset() before: it makes replays drift further.
        """
        self.env.sim.set_state_from_flattened(np.asarray(sim_state))
        self.env.sim.forward()
        for controller in self.env.robots[
            0
        ].composite_controller.part_controllers.values():
            controller.update(force=True)
            controller.reset_goal()
        return self.env._get_observations(force_update=True)

    def success(self):
        """The task's own success test (Lift: cube 4 cm above the table)"""
        return bool(self.env._check_success())

    def grasped(self):
        """The gripper holds one of the task's objects (robosuite's contact test)"""
        objects = [self.env.cube] if hasattr(self.env, "cube") else self.env.objects
        gripper = self.env.robots[0].gripper
        return any(self.env._check_grasp(gripper=gripper, object_geoms=o) for o in objects)

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
