import time

from duobench import SCENE_PATHS
import numpy as np
from rcs import CAMERA_PATHS, DEFAULT_TRANSFORMS, OBJECT_PATHS
from rcs._core import common
from rcs._core.sim import CameraType, SimCameraConfig
from rcs.envs.configs import EmptyWorldDroid
from rcs.envs.scenes import CameraAdderConfig, SimEnvCreatorConfig


class SingleArm(EmptyWorldDroid):
    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.scene = SCENE_PATHS["single_arm"]
        # 0.011 = width of mount plate, 0.035=distance from center of mounting plate to franka
        # workstation depth / 2 - front depth + distance to plate back + plate depth / 2
        # >>> 81.0/2 - 31.5+4.5 + 25.5/2
        # 26.25
        cfg.root_frame_to_world = common.Pose(translation=[0.2625+0.035, 0, 0.7533+0.011])

        cfg.root_frame_objects = {
            "fr3_single_mount": (OBJECT_PATHS["fr3_single_mount"], common.Pose(translation=[-0.035, 0, -0.011], rpy_vector=[0, 0, -np.deg2rad(90)])),
        }
        cfg.robot_frame_objects = {
            "right": {
                "zed_mount": (
                    OBJECT_PATHS["droid_wrist_mount"],
                    common.Pose(rpy_vector=[-np.pi/2, 0, np.pi/2], translation=[-0.034, 0, -0.008]),
                )
            },
        }
        cfg.camera_adds["front"] = CameraAdderConfig(
            xml_path=CAMERA_PATHS["zed2i"],
            offset=common.Pose(
                translation=[0.958473, 0.000714, 0.499707],
                rpy_vector=[2.530727947461584, 0.0, np.pi / 2],
            ),
        )
        cfg.camera_cfgs = {
            "front": SimCameraConfig(
                identifier="front",
                type=CameraType.fixed,
                resolution_width=1280,
                resolution_height=720,
                frame_rate=30,
            ),
            "wrist": SimCameraConfig(
                identifier="wrist",
                type=CameraType.fixed,
                resolution_width=1280,
                resolution_height=720,
                frame_rate=30,
            ),
        }
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
