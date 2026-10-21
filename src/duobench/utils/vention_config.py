import copy
import time

import gymnasium as gym
from rcs import DEFAULT_TRANSFORMS, SCENE_PATHS
from rcs.envs.configs import EmptyWorldFR3Duo
from rcs.envs.scenes import MjModel, SimEnvCreatorConfig

# MuJoCo sizes the arena from the scene at rest (15-22 MiB for the DuoBench scenes), contact-rich episodes
# (e.g. ball_maze with ~170 contacts) exceed it and abort with "mj_stackAlloc: insufficient memory"
MUJOCO_MEMORY = 256 * 1024 * 1024


class VentionSceneFR3Duo(EmptyWorldFR3Duo):
    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.scene = SCENE_PATHS["vention_world"]
        cfg.root_frame_to_world = DEFAULT_TRANSFORMS["VENTION_HEIGHT_OFFSET"]
        cfg.sim_cfg.memory = MUJOCO_MEMORY
        return cfg

    def create_env_from_model(self, cfg: SimEnvCreatorConfig, mjmodel: MjModel) -> gym.Env:
        # eval setups often replace sim_cfg with a fresh SimConfig, keep the larger arena unless set explicitly
        if cfg.sim_cfg.memory < 0:
            cfg = copy.copy(cfg)
            cfg.sim_cfg = copy.copy(cfg.sim_cfg)
            cfg.sim_cfg.memory = MUJOCO_MEMORY
        return super().create_env_from_model(cfg, mjmodel)


if __name__ == "__main__":
    scene = VentionSceneFR3Duo()
    env = scene.create_env(scene.config())
    obs, info = env.reset()
    print(obs)
    # Duo
    for _ in range(100):
        for _ in range(10):
            # move 1cm in x direction (forward) and close gripper
            act = {
                "left": {"tquat": [0.01, 0, 0, 0, 0, 0, 1], "gripper": [0]},
                "right": {"tquat": [0.01, 0, 0, 0, 0, 0, 1], "gripper": [0]},
            }
            obs, reward, terminated, truncated, info = env.step(act)
            # print(obs)
            print(reward, terminated, truncated, info)
            time.sleep(0.033)
        for _ in range(10):
            # move 1cm in negative x direction (backward) and open gripper
            act = {
                "left": {"tquat": [-0.01, 0, 0, 0, 0, 0, 1], "gripper": [1]},
                "right": {"tquat": [-0.01, 0, 0, 0, 0, 0, 1], "gripper": [1]},
            }
            obs, reward, terminated, truncated, info = env.step(act)
            # print(obs)
            print(reward, terminated, truncated, info)
            time.sleep(0.033)
