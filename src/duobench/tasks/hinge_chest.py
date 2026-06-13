from dataclasses import dataclass, field

import gymnasium as gym
import mujoco as mj
import numpy as np
import rcs
from rcs.envs.scenes import BaseTaskConfig, SimEnvCreatorConfig, Task
from rcs.sim.composer import ModelComposer
from rcs.sim.sim import Sim

from duobench.tasks import TaskStage, TaskStageWrapper
from duobench.utils.helper_wrappers import (
    RandomSquareObjsPos,
    body_pose_in_site_frame,
    get_bodies_in_contact_with_gripper_pad,
)
from duobench.utils.vention_config import VentionSceneFR3Duo


class HingeChestStage(TaskStage):
    INSTRUCTION = "open the box with the right arm and place the cube inside the box with the left arm"

    def __init__(self, goal_site, target_body, door_joint, goal_site_dimensions, prefix):
        super().__init__(
            max_stage=3,
            internal_state={
                "object_picked": False,
                "door_open": False,
                "object_placed": False,
            },
            stage_to_subinstructions={
                0: "open the chest OR pick up the box",
                1: "pick up the box AND open the chest",
                2: "place the box inside the chest",
                3: "task completed; the box is inside the chest",
            },
            instruction=self.INSTRUCTION,
        )
        self.goal_site = goal_site
        self.target_body = target_body
        self.door_joint = door_joint
        self.goal_site_dimensions = goal_site_dimensions
        self.prefix = prefix

    def update_internal_state(self, sim):
        goal_site_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_SITE, self.prefix + self.goal_site)
        target_body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, self.prefix + self.target_body + "_body")
        door_joint_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_JOINT, self.prefix + self.door_joint)
        door_joint_pos = sim.data.qpos[sim.model.jnt_qposadr[door_joint_id]]

        # Check if object is picked
        object_picked = False
        left_box_picked_bodies = get_bodies_in_contact_with_gripper_pad(sim, "left")
        if target_body_id in left_box_picked_bodies:
            object_picked = True
        right_box_picked_bodies = get_bodies_in_contact_with_gripper_pad(sim, "right")
        if target_body_id in right_box_picked_bodies:
            object_picked = True
        self.internal_state["object_picked"] = object_picked or self.internal_state["object_placed"]

        # Check if door is open
        self.internal_state["door_open"] = (door_joint_pos < -0.8) or self.internal_state["object_placed"]

        gs_target_body_T = body_pose_in_site_frame(sim.data, target_body_id, goal_site_id)
        gs_target_body_pos = gs_target_body_T[:3, 3]

        # Check if the target object is within the goal site region
        target_within_goal_site_x = abs(gs_target_body_pos[0]) <= self.goal_site_dimensions[0]
        target_within_goal_site_y = abs(gs_target_body_pos[1]) <= self.goal_site_dimensions[1]
        # z should only consider that the object is at most goal_site_dimensions[2] above the site
        target_within_goal_site_z = abs(gs_target_body_pos[2]) <= self.goal_site_dimensions[2]
        self.internal_state["object_placed"] = (
            target_within_goal_site_x and target_within_goal_site_y and target_within_goal_site_z
        )
        self.update_stage()

    def update_stage(self):
        do = self.internal_state["door_open"]
        opl = self.internal_state["object_placed"]
        opk = self.internal_state["object_picked"]

        if opl:
            new_stage = self.max_stage
        elif do and opk:
            new_stage = 2
        elif (do and not opk) or (opk and not do):  # only one of the two subgoals is completed
            new_stage = 1
        else:
            new_stage = 0

        self.stage = max(self.stage, new_stage)


"""
Success condition for the task:
Initial phase:
The door needs to be opened, i.e. the microwave joint needs to be < -0.8 or so, and remain that way until the final phase is finished


Final phase:
The pose of the cup must be within the range of the site, in a rectangle, in the site's frame, of shape
x len +-0.080 
y len +-0.095 
z height 0.04
while the z-axis of the mug needs to be within 20 degrees of the z-axis of the site (cosine similarity should be enough)

"""


@dataclass(kw_only=True)
class HingeChestTaskConfig(BaseTaskConfig):

    z_init = 0.02
    task_id: str = "hinge_chest"
    bowl_radius = 0.06

    x_width = 0.35
    y_width = 0.35
    obj_position_margin = 0.08
    prefix = "HingeChest_"
    include_rotation = True
    objects_xml: dict[str, str] = field(
        default_factory=lambda: {
            "black_box": rcs.OBJECT_PATHS["parallel_pick_black_box"],
        }
    )
    objects_joints: dict[str, str] = field(
        default_factory=lambda: {
            "black_box": "black_box_joint",
        }
    )

    object_center_to_root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(
            # 0.02 = z_init
            translation=np.array([0.5, 0.2, 0.02]),
            quaternion=np.array([0, 0, 0, 1]),
        )
    )
    goals_xml: dict[str, str] = field(default_factory=lambda: {"hinge_chest": rcs.OBJECT_PATHS["hinge_chest_chest"]})

    goals_center_to_root_frame: dict[str, rcs.common.Pose] = field(
        default_factory=lambda: {
            # 0.02 = z_init
            "hinge_chest": rcs.common.Pose(translation=np.array([0.6, -0.2, 0.02]), quaternion=np.array([0, 0, 1, 0])),
        }
    )
    goal_site = "placement_site"
    door_joint = "chest_hinge"
    goal_site_dimensions: np.ndarray = field(default_factory=lambda: np.array([0.065, 0.115, 0.03]))


class HingeChestTask(Task[HingeChestTaskConfig]):
    @staticmethod
    def add_task_mujoco(cfg: HingeChestTaskConfig, composer: ModelComposer, env_cfg: SimEnvCreatorConfig):
        """Add task-specific elements to the Mujoco scene."""
        object2world = cfg.object_center_to_root_frame * env_cfg.root_frame_to_world

        for xml in cfg.objects_xml.values():
            composer.add_object_world_frame(
                xml, object_prefix=cfg.prefix, pose=object2world, register_root_relative_replay_free_joints=True
            )

        for goal_name, xml in cfg.goals_xml.items():
            center = cfg.goals_center_to_root_frame[goal_name]
            goal2world = center * env_cfg.root_frame_to_world
            composer.add_object_world_frame(
                xml, object_prefix=cfg.prefix, pose=goal2world, register_root_relative_replay_free_joints=True
            )

    @staticmethod
    def add_task_env(cfg: HingeChestTaskConfig, env: gym.Env, simulation: Sim, env_cfg: SimEnvCreatorConfig) -> gym.Env:
        """Add task-specific wrappers to the environment."""
        _ = simulation
        object2world = env_cfg.root_frame_to_world * cfg.object_center_to_root_frame
        obj_joint_names = [cfg.prefix + joint for joint in cfg.objects_joints.values()]

        # For positioning target objects
        env = RandomSquareObjsPos(
            env,
            x_width=cfg.x_width,
            y_width=cfg.y_width,
            z_init=cfg.z_init,
            center2world=object2world,
            include_rotation=cfg.include_rotation,
            obj_joint_names=obj_joint_names,
            obj_position_margin=cfg.obj_position_margin,
        )

        # # For positioning the bowls
        # for k, v in cfg.goals_center_to_root_frame.items():
        #     goal2world = v * env_cfg.root_frame_to_world
        #     env = RandomSquareObjsPos(
        #         env,
        #         x_width=0.1,
        #         y_width=0.1,
        #         z_init=cfg.z_init,
        #         center2world=goal2world,
        #         include_rotation=False,
        #         obj_joint_names=[cfg.prefix + k + "_freejoint"],
        #     )
        return TaskStageWrapper(
            env,
            HingeChestStage(
                goal_site=cfg.goal_site,
                target_body=next(iter(cfg.objects_xml)),
                door_joint=cfg.door_joint,
                goal_site_dimensions=cfg.goal_site_dimensions,
                prefix=cfg.prefix,
            ),
        )


rcs.TASKS["hinge_chest"] = HingeChestTask


class HingeChestEnvConfig(VentionSceneFR3Duo):

    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.task_cfg = HingeChestTaskConfig()
        return cfg


gym.register(id="duobench/hinge_chest", entry_point=HingeChestEnvConfig())
