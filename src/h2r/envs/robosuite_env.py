"""robosuite environment"""

import mujoco
import numpy as np
import robosuite as suite
from scipy.spatial.transform import Rotation


# Gap between the two finger joints (qpos[0] - qpos[1]) fully open and fully closed on nothing, for the closedness in
# [0, 1]. Panda: the joint ranges (0.08, 0), as batgre/h2r-sim-v3 was collected. Sawyer: measured on 10 Nov. 2026
# (open command: 0.0113 / -0.0113, close: -0.0118 / 0.0118; its joint ranges are not the open and closed positions)
FINGER_GAP = {"PandaGripper": (0.08, 0.0), "RethinkGripper": (0.0226, -0.0236)}


class RobosuiteEnv:
    def __init__(self, cfg: dict):
        """has_renderer: the MuJoCo window (mjviewer, under mjpython on macOS). robosuite's offscreen renderer is then
        off (on macOS it opens a GLFW window, which mjpython's thread cannot) and frame() renders with mujoco's
        Renderer, with robosuite's options: the same pixels"""
        self.cfg = dict(cfg)
        if self.cfg.get("has_renderer"):
            self.cfg["has_offscreen_renderer"] = False
        self.env = suite.make(**self.cfg)
        self.renderer = None

    def reset(self, seed=None):
        """seed: robosuite places the objects with numpy's global generator, so the same seed as the collection gives
        the same scene as the recorded episode"""
        if seed is not None:
            np.random.seed(seed)
        return self.env.reset()

    def frame(self):
        """(H, W, 3) uint8 image of the first camera, upright (robosuite renders upside down)"""
        names = self.cfg["camera_names"]
        camera = names if isinstance(names, str) else names[0]
        height, width = self.cfg["camera_heights"], self.cfg["camera_widths"]
        if self.cfg.get("has_offscreen_renderer", True):
            return self.env.sim.render(camera_name=camera, height=height, width=width)[::-1].copy()
        model = self.env.sim.model._model
        if self.renderer is None or self.renderer.model is not model:  # a reset rebuilds the model (hard reset)
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = mujoco.Renderer(model, height, width, max_geom=10000)
        option = mujoco.MjvOption()
        option.geomgroup[0] = 0  # as robosuite: no collision meshes
        self.renderer.update_scene(self.env.sim.data._data, camera=camera, scene_option=option)
        return self.renderer.render().copy()

    def grasping(self):
        """both finger pads touch the object (robosuite's grasp test); Lift: the cube"""
        return self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.cube)

    def table_height(self):
        return self.env.model.mujoco_arena.table_offset[2]

    def sim_state(self):
        """MuJoCo state, then the gripper's command integrator (robosuite keeps it outside the MuJoCo state)"""
        gripper = next(iter(self.env.robots[0].gripper.values()))
        return np.r_[self.env.sim.get_state().flatten(), gripper.current_action]

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

        gripper = type(next(iter(robot.gripper.values()))).__name__
        if gripper not in FINGER_GAP:
            raise NotImplementedError(f"no closedness reading for {gripper} (see FINGER_GAP)")
        opened, closed = FINGER_GAP[gripper]
        q = obs["robot0_gripper_qpos"]
        return np.r_[pos, rot, np.clip((opened - (q[0] - q[1])) / (opened - closed), 0, 1)]


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
