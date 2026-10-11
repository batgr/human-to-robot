import numpy as np
import robosuite as suite


class RobosuiteEnv:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.env = suite.make(**self.cfg)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.cfg.get("has_renderer", False):
            self.env.render()
        return obs, reward, done, info


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
