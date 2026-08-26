import time

from duobench import SCENE_PATHS
from rcs import DEFAULT_TRANSFORMS
from rcs._core import common
from rcs.envs.configs import EmptyWorldDroid, EmptyWorldFR3, EmptyWorldFR3Duo
from rcs.envs.scenes import SimEnvCreatorConfig


class SingleArm(EmptyWorldDroid):
    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.scene = SCENE_PATHS["single_arm"]
        cfg.root_frame_to_world = common.Pose()
        return cfg


if __name__ == "__main__":
    scene = SingleArm()
    env = scene.create_env(scene.config())
    obs, info = env.reset()
    print(obs)
    # Duo
    for _ in range(100):
        for _ in range(10):
            # move 1cm in x direction (forward) and close gripper
            act = {
                "right": {"tquat": [0.01, 0, 0, 0, 0, 0, 1], "gripper": [0]},
            }
            obs, reward, terminated, truncated, info = env.step(act)
            # print(obs)
            print(reward, terminated, truncated, info)
            time.sleep(0.033)
        for _ in range(10):
            # move 1cm in negative x direction (backward) and open gripper
            act = {
                "right": {"tquat": [-0.01, 0, 0, 0, 0, 0, 1], "gripper": [1]},
            }
            obs, reward, terminated, truncated, info = env.step(act)
            # print(obs)
            print(reward, terminated, truncated, info)
            time.sleep(0.033)
