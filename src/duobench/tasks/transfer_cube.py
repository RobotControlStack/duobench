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
    get_bodies_in_contact_with_gripper_pad,
    get_geom_pos,
)
from duobench.utils.vention_config import VentionSceneFR3Duo


class TransferCubeStage(TaskStage):
    def __init__(
        self,
        cfg: "TransferCubeTaskConfig",
        goals: dict[str, list[str]],
        bowl_radius: float = 0.06,
        prefix: str = "TransferAndPlace_",
    ):
        super().__init__(
            max_stage=4,
            internal_state={
                "grasped_and_lifted": False,
                "both_grippers_contact": False,
                "successfully_transferred": False,
                "correctly_placed": False,
                "arms_retracted": False,
            },
            stage_to_subinstructions={
                0: "pick up the cube",
                1: "bring both grippers into contact with the cube",
                2: "transfer the cube to the other gripper",
                3: "place the cube correctly and release it",
                4: "task completed; the cube is placed and the arms are retracted",
            },
            instruction=cfg.task_instructions,
        )
        self.goals = goals
        self.bowl_radius = bowl_radius
        self.prefix = prefix
        self.cfg = cfg
        self.initial_holder: str | None = None

    def reset(self):
        super().reset()
        self.initial_holder = None

    def update_internal_state(self, sim: Sim):
        goal_name = next(iter(self.goals))
        target_name = self.goals[goal_name][0]
        goal_pos = get_geom_pos(sim.model, sim.data, self.prefix + goal_name + "_visual_geom")
        cube_pose = get_geom_pos(sim.model, sim.data, self.prefix + target_name + "_visual_geom")
        target_body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, self.prefix + target_name + "_body")
        left_contacts = set(get_bodies_in_contact_with_gripper_pad(sim, "left"))
        right_contacts = set(get_bodies_in_contact_with_gripper_pad(sim, "right"))
        left_gripper_contact = target_body_id in left_contacts
        right_gripper_contact = target_body_id in right_contacts

        # check first step
        lifted = cube_pose[2] > self.cfg.table_hight + 0.012
        grasped_and_lifted = lifted and (left_gripper_contact or right_gripper_contact)

        both_grippers_contact = left_gripper_contact and right_gripper_contact

        sole_holder = None
        if left_gripper_contact and not right_gripper_contact:
            sole_holder = "left"
        elif right_gripper_contact and not left_gripper_contact:
            sole_holder = "right"

        if (
            lifted
            and sole_holder is not None
            and not self.internal_state["both_grippers_contact"]
            and self.initial_holder is None
        ):
            self.initial_holder = sole_holder

        successfully_transferred = (
            self.internal_state["both_grippers_contact"]
            and lifted
            and sole_holder is not None
            and self.initial_holder is not None
            and sole_holder != self.initial_holder
        )

        goal_dist = float(np.linalg.norm(goal_pos - cube_pose))
        correctly_placed = bool(goal_dist < self.bowl_radius)
        arms_retracted = bool(correctly_placed and not left_gripper_contact and not right_gripper_contact)

        self.internal_state["grasped_and_lifted"] = self.internal_state["grasped_and_lifted"] or grasped_and_lifted
        self.internal_state["both_grippers_contact"] = (
            self.internal_state["both_grippers_contact"] or both_grippers_contact
        )
        self.internal_state["successfully_transferred"] = (
            self.internal_state["successfully_transferred"] or successfully_transferred
        )
        self.internal_state["correctly_placed"] = bool(self.internal_state["correctly_placed"] or correctly_placed)
        self.internal_state["arms_retracted"] = bool(arms_retracted)

        self.update_stage()

    def update_stage(self):
        if self.internal_state["correctly_placed"] and self.internal_state["arms_retracted"]:
            self.stage = 4
        elif self.internal_state["successfully_transferred"]:
            self.stage = 3
        elif self.internal_state["both_grippers_contact"]:
            self.stage = 2
        elif self.internal_state["grasped_and_lifted"]:
            self.stage = 1
        else:
            self.stage = 0


@dataclass(kw_only=True)
class TransferCubeTaskConfig(BaseTaskConfig):

    z_init = 0.02
    task_id: str = "transfer_cube"
    bowl_radius = 0.06
    task_instructions = "grasp the white cube with the right arm, hand it over to the left arm and place it in the white bowl with the left arm"
    x_width = 0.35
    y_width = 0.35
    obj_position_margin = 0.08
    prefix = "TransferAndPlace_"
    include_rotation = True
    objects_xml: dict[str, str] = field(
        default_factory=lambda: {
            "white_long_box": rcs.OBJECT_PATHS["handover_hole_white_long_box"],
        }
    )
    objects_joints: dict[str, str] = field(
        default_factory=lambda: {
            "white_long_box": "white_long_box_joint",
        }
    )
    object_center_to_root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(
            # 0.02 = z_init
            translation=np.array([0.5, -0.2, 0.02]),
            quaternion=np.array([0, 0, 0, 1]),
        )
    )
    goals_xml: dict[str, str] = field(
        default_factory=lambda: {"white_bowl": rcs.OBJECT_PATHS["parallel_pick_white_bowl"]}
    )

    goals_objects: dict[str, list[str]] = field(
        default_factory=lambda: {
            "white_bowl": [
                # "big_red_cube",
                "white_long_box",
            ]
        }
    )

    goals_center_to_root_frame: dict[str, rcs.common.Pose] = field(
        default_factory=lambda: {
            # 0.02 = z_init
            "white_bowl": rcs.common.Pose(translation=np.array([0.6, 0.3, 0.02]), quaternion=np.array([0, 0, 0, 1])),
            # 0.02 = z_init
        }
    )

    table_hight = 1


class TransferCubeTask(Task[TransferCubeTaskConfig]):
    @staticmethod
    def add_task_mujoco(cfg: TransferCubeTaskConfig, composer: ModelComposer, env_cfg: SimEnvCreatorConfig):
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
    def add_task_env(
        cfg: TransferCubeTaskConfig, env: gym.Env, simulation: Sim, env_cfg: SimEnvCreatorConfig
    ) -> gym.Env:
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
            flip_x=False,
        )

        # For positioning the bowls
        for k, v in cfg.goals_center_to_root_frame.items():
            goal2world = v * env_cfg.root_frame_to_world
            env = RandomSquareObjsPos(
                env,
                x_width=0.1,
                y_width=0.1,
                z_init=cfg.z_init,
                center2world=goal2world,
                include_rotation=False,
                obj_joint_names=[cfg.prefix + k + "_joint"],
            )
        return TaskStageWrapper(
            env, TransferCubeStage(cfg, cfg.goals_objects, bowl_radius=cfg.bowl_radius, prefix=cfg.prefix)
        )


rcs.TASKS["transfer_cube"] = TransferCubeTask


class TransferCubeEnvConfig(VentionSceneFR3Duo):

    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.task_cfg = TransferCubeTaskConfig()
        return cfg


gym.register(id="duobench/transfer_cube", entry_point=TransferCubeEnvConfig())
