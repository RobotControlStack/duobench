import time

from duobench.tasks.flipping import FlippingEnvConfig
from rcs._core.sim import SimConfig
from rcs.envs.base import ControlMode, RelativeTo


if __name__ == "__main__":
    scene = FlippingEnvConfig()
    cfg = scene.config()
    cfg.control_mode = ControlMode.JOINTS
    cfg.relative_to = RelativeTo.NONE
    cfg.headless = True
    cfg.sim_cfg = SimConfig(
        async_control=True, realtime=True, frequency=25, max_convergence_steps=500
    )
    env = scene.create_env(cfg)
    obs, info = env.reset()
    print(obs)

    for _ in range(100):
        obs, info = env.reset()
        for _ in range(10):
            # sample random relative action and execute it
            act = env.action_space.sample()
            print(act)
            obs, reward, terminated, truncated, info = env.step(act)