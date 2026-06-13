import logging
from time import sleep
from typing import Any, cast

import gymnasium as gym
import mujoco
import numpy as np
from rcs._core.common import Pose, GripperType
from rcs._core.sim import SimGripperConfig, SimRobot
from rcs.envs.base import ControlMode, GripperWrapper, RelativeTo
from rcs.envs.configs import EmptyWorldFR3

import rcs
import duobench

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PickUpDemo:
    num_waypoints = 60
    def __init__(self, env: gym.Env):
        self.env = env
        self._robot = cast(SimRobot, self.env.get_wrapper_attr("robot"))["robot"]
        self.home_pose = self._robot.get_cartesian_position()

    def _action(self, pose: Pose, gripper: list[float]) -> dict[str, Any]:
        return {"robot":{"xyzrpy": pose.xyzrpy(), "gripper": [gripper]}}

    def get_object_pose(self, geom_name, geom=True) -> Pose:
        model = self.env.get_wrapper_attr("sim").model
        data = self.env.get_wrapper_attr("sim").data

        if geom:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            obj_pose_world_coordinates = Pose(
                translation=data.geom_xpos[geom_id], rotation=data.geom_xmat[geom_id].reshape(3, 3)
            ) * Pose(rpy_vector=np.array([0, 0, 0]), translation=np.array([0.0, 0.0, 0.0]))
        else:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, geom_name)
            obj_pose_world_coordinates = Pose(
                translation=data.xpos[body_id], rotation=data.xmat[body_id].reshape(3, 3)
            ) * Pose(rpy_vector=np.array([0, 0, 0]), translation=np.array([0.0, 0.0, 0.0]))
            
        return self._robot.to_pose_in_robot_coordinates(obj_pose_world_coordinates)

    def generate_waypoints(self, start_pose: Pose, end_pose: Pose, num_waypoints: int) -> list[Pose]:
        waypoints = []
        for i in range(num_waypoints + 1):
            t = i / (num_waypoints)
            waypoints.append(start_pose.interpolate(end_pose, t))
        return waypoints

    def step(self, action: dict) -> dict:
        return self.env.step(action)[0]

    def plan_linear_motion(self, geom_name: str, delta_up: float, num_waypoints: int = 20) -> list[Pose]:
        end_eff_pose = self._robot.get_cartesian_position()
        goal_pose = self.get_object_pose(geom_name=geom_name, geom=True)
        goal_pose *= Pose(translation=np.array([0.0, 0.0, delta_up]), quaternion=np.array([1, 0, 0, 0]))  # type: ignore
        return self.generate_waypoints(end_eff_pose, goal_pose, num_waypoints=num_waypoints)

    def execute_motion(self, waypoints: list[Pose], gripper: list[float] = GripperWrapper.BINARY_GRIPPER_OPEN) -> dict:
        obs = {}
        for i in range(len(waypoints)):
            obs = self.step(self._action(waypoints[i], gripper))
        return obs

    def approach(self, geom_name: str):
        waypoints = self.plan_linear_motion(geom_name=geom_name, delta_up=0.2, num_waypoints=10)
        self.execute_motion(waypoints=waypoints, gripper=GripperWrapper.BINARY_GRIPPER_OPEN)

    def grasp(self, geom_name: str):

        waypoints = self.plan_linear_motion(geom_name=geom_name, delta_up=0.02, num_waypoints=self.num_waypoints)
        self.execute_motion(waypoints=waypoints, gripper=GripperWrapper.BINARY_GRIPPER_OPEN)

        for _ in range(4):
            self.step(self._action(self._robot.get_cartesian_position(), GripperWrapper.BINARY_GRIPPER_CLOSED))

        waypoints = self.plan_linear_motion(geom_name=geom_name, delta_up=0.2, num_waypoints=self.num_waypoints)
        self.execute_motion(waypoints=waypoints, gripper=GripperWrapper.BINARY_GRIPPER_CLOSED)

    def move_home(self):
        end_eff_pose = self._robot.get_cartesian_position()
        waypoints = self.generate_waypoints(end_eff_pose, self.home_pose, num_waypoints=self.num_waypoints)
        self.execute_motion(waypoints=waypoints, gripper=GripperWrapper.BINARY_GRIPPER_CLOSED)

    def pickup(self, geom_name: str):
        self.approach(geom_name)
        self.grasp(geom_name)
        self.move_home()


def main():
    scene = EmptyWorldFR3()
    cfg = scene.config()
    cfg.control_mode = ControlMode.CARTESIAN_TRPY
    cfg.robot_cfgs['robot'].tcp_offset = rcs.GRIPPER_OFFSETS[rcs.common.GripperType("Robotiq2F85")]
    cfg.relative_to = RelativeTo.NONE
    gripper_cfg = SimGripperConfig(
        epsilon_inner=0.005,
        epsilon_outer=0.005,
        seconds_between_callbacks=0.1,
        ignored_collision_geoms=[],
        collision_geoms=[],
        collision_geoms_fingers=[],
        joints=["right_driver_joint", "left_driver_joint"],
        max_joint_width=0.005,
        min_joint_width=1.0,
        actuator="fingers_actuator",
        max_actuator_width=0,
        min_actuator_width=255,
        gripper_type=GripperType("Robotiq2F85"),
    )
    cfg.gripper_cfgs = {
        "robot": gripper_cfg
    }
    gripper_offset = rcs.common.Pose(
        quaternion=np.array([0, 0, 0.7071068, 0.7071068]), translation=np.array([0.0, 0.0, 0.0])
    )
    cfg.gripper_offsets = {"robot": gripper_offset}
    cfg.sim_cfg.realtime = False
    cfg.sim_cfg.async_control = True
    cfg.max_relative_movement = None
    cfg.root_frame_objects = {
        "green_cube": (
            rcs.OBJECT_PATHS["parallel_pick_black_tri_cylinder"],
            # rcs.OBJECT_PATHS["spring_door_simple_mug"],
            Pose(translation=np.array([0.5, 0.0, 0.05]), quaternion=np.array([0.0, 0.0, 1, 0])),
        )
    }
    env = scene.create_env(cfg)
    env.get_wrapper_attr("sim").open_gui()
    # wait for gui to open
    # sleep(3)
    for _ in range(100):
        env.reset()
        # print(env.get_wrapper_attr("robot").get_cartesian_position().translation())  # type: ignore
        controller = PickUpDemo(env)
        # controller.pickup("green_cube_mug_body")
        controller.pickup("box_geom")


if __name__ == "__main__":
    main()
