import time

from duobench import SCENE_PATHS
import numpy as np
from rcs import CAMERA_PATHS, DEFAULT_TRANSFORMS, OBJECT_PATHS
from rcs._core import common
from rcs._core.sim import CameraType, SimCameraConfig, SimConfig
from rcs.envs.base import ControlMode, RelativeTo
from rcs.envs.configs import EmptyWorldDroid
from rcs.envs.scenes import CameraAdderConfig, SimEnvCreatorConfig

# Camera extrinsics from hand-eye calibration, expressed as the composer offset:
# the camera-body pose in its attach frame in MuJoCo camera convention (looks -Z, +Y up).
# Built as  offset = T_attach_cam(opencv) @ Rx(180deg) @ inv(<camera> local mount pose),
# so offset @ (XML <camera> local pose) reproduces the measured optical pose.
# Wrist: attach frame is the FR3 attachment_site (flange), with TCP = site + z*0.1493.
_WRIST_CAM_OFFSET = np.array(
    [
        [0.322504, 0.017714, -0.946402, -0.071548],
        [0.014907, 0.999606, 0.023790, 0.007081],
        [0.946451, -0.021781, 0.322113, 0.021495],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
# Front: attach frame is the robot base (== root_frame_to_world, no rotation).
_FRONT_CAM_OFFSET = np.array(
    [
        [0.033404, 0.777963, 0.627422, 0.943064],
        [0.999353, -0.034364, -0.010597, 0.070509],
        [0.013317, 0.627370, -0.778607, 0.481027],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class SingleArm(EmptyWorldDroid):
    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.scene = SCENE_PATHS["single_arm"]
        # 0.011 = width of mount plate, 0.035=distance from center of mounting plate to franka
        # workstation depth / 2 - front depth + distance to plate back + plate depth / 2
        # >>> 81.0/2 - 31.5+4.5 + 25.5/2
        # 26.25
        cfg.root_frame_to_world = common.Pose(translation=[0.2625+0.035, 0, 0.7533+0.011])
        # cfg.robot_cfgs["right"].kp = [100., 100., 100., 100., 75., 150., 50.]
        # cfg.robot_cfgs["right"].kv = [20., 20., 20., 20., 7.5, 15.0, 5.0]

        # cfg.robot_cfgs["right"].kp = np.array([600, 600, 600, 600, 250, 150, 50])
        # cfg.robot_cfgs["right"].kv = np.array([50, 50, 50, 50, 30, 25, 15])

        cfg.control_mode = ControlMode.JOINTS
        cfg.relative_to = RelativeTo.NONE
        # cfg.headless = True
        cfg.sim_cfg = SimConfig(
            async_control=True, realtime=True, frequency=25, max_convergence_steps=500
        )

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
        cfg.camera_adds["front_rgb"] = CameraAdderConfig(
            xml_path=CAMERA_PATHS["zed2i"],
            offset=common.Pose(pose_matrix=_FRONT_CAM_OFFSET),  # hand-eye calibrated (base frame)
        )
        cfg.camera_adds.pop("wrist", None)
        cfg.camera_adds["gripper_rgb"] = CameraAdderConfig(
            xml_path=CAMERA_PATHS["zed_mini"],
            offset=common.Pose(pose_matrix=_WRIST_CAM_OFFSET),  # hand-eye calibrated (attachment_site frame)
            robot_name="right",
        )
        cfg.gravcomp_ignore = (cfg.gravcomp_ignore - {"wrist"}) | {"gripper_rgb"}
        cfg.camera_cfgs = {
            "front_rgb": SimCameraConfig(
                identifier="front_rgb",
                type=CameraType.fixed,
                resolution_width=1280,
                resolution_height=720,
                frame_rate=30,
            ),
            "gripper_rgb": SimCameraConfig(
                identifier="gripper_rgb",
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
