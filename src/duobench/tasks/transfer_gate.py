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
    check_contact_between_bodies,
    get_bodies_in_contact_with_gripper_pad,
)
from duobench.utils.vention_config import VentionSceneFR3Duo


class TransferGateStage(TaskStage):
    """
    TaskStage to check whether the open spring door and place object inside task is successful.
    Given the goal site's name and the target object body's name, it performs the following checks at every step:
    - whether the door is open enough (microwave joint < -0.8)
    - whether the target object is within a certain distance threshold to the goal site,
    - and whether the target object is oriented in a similar way as the goal site (z-axis of mug within 20 degrees of z-axis of site).
    """

    INSTRUCTION = "use the right arm to pick up the white box, and hand it over to the left arm through the hoop, then place it on the green mat with the left arm"

    def __init__(self, target_body, ring_site, mat_body, mat_dimensions, prefix):
        super().__init__(
            max_stage=4,
            internal_state={
                "object_picked": False,
                "object_passed_ring": False,
                "object_handover": False,
                "object_on_mat": False,
            },
            stage_to_subinstructions={
                0: "pick up the box",
                1: "pass the box through the ring",
                2: "grab the box with the other hand",
                3: "place the box on the mat",
                4: "task completed; the box is on the mat",
            },
            instruction=self.INSTRUCTION,
        )

        self.target_body = target_body
        self.ring_site = ring_site
        self.mat_body = mat_body
        self.mat_dimensions = mat_dimensions
        self.prefix = prefix

    def update_internal_state(self, sim):
        target_body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, self.prefix + self.target_body + "_body")
        ring_site_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_SITE, self.prefix + self.ring_site)
        mat_body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, self.prefix + self.mat_body)

        # Check if object is picked
        object_picked = False
        right_box_picked_bodies = get_bodies_in_contact_with_gripper_pad(sim, "right")
        if target_body_id in right_box_picked_bodies:
            object_picked = True
        self.internal_state["object_picked"] = object_picked or self.internal_state["object_passed_ring"]

        # Check if object passed ring
        ring_target_body_T = body_pose_in_site_frame(sim.data, target_body_id, ring_site_id)
        ring_target_body_pos = ring_target_body_T[:3, 3]
        self.internal_state["object_passed_ring"] = (
            np.sqrt(ring_target_body_pos[0] ** 2 + ring_target_body_pos[2] ** 2) <= 0.11
            and abs(ring_target_body_pos[1]) <= 0.05
        ) or self.internal_state["object_passed_ring"]

        # Check if handover is successful
        object_handover = False
        left_box_picked_bodies = get_bodies_in_contact_with_gripper_pad(sim, "left")
        if target_body_id in left_box_picked_bodies:
            object_handover = True
        self.internal_state["object_handover"] = object_handover or self.internal_state["object_on_mat"]

        # Check if object is on mat
        self.internal_state["object_on_mat"] = (
            check_contact_between_bodies(sim, target_body_id, mat_body_id) or self.internal_state["object_on_mat"]
        )
        self.update_stage()

    def update_stage(self):
        if self.internal_state["object_on_mat"]:
            self.stage = 4
        elif self.internal_state["object_handover"]:
            self.stage = 3
        elif self.internal_state["object_passed_ring"]:
            self.stage = 2
        elif self.internal_state["object_picked"]:
            self.stage = 1
        else:
            self.stage = 0


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
class TransferGateTaskConfig(BaseTaskConfig):

    z_init = 0.02
    task_id: str = "transfer_gate"
    mat_x_width = 0.1
    mat_y_width = 0.1

    obj_x_width = 0.35
    obj_y_width = 0.35

    stand_x_width = 0.05
    stand_y_width = 0.1

    obj_position_margin = 0.08
    prefix = "TransferGateTask_"
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

    mat_xml: dict[str, str] = field(
        default_factory=lambda: {
            "mat": rcs.OBJECT_PATHS["handover_hole_mat"],
        }
    )

    stand_xml: dict[str, str] = field(
        default_factory=lambda: {
            "stand": rcs.OBJECT_PATHS["handover_hole_stand"],
        }
    )

    object_center_to_root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(
            translation=np.array([0.5, -0.3, 0.02]), quaternion=np.array([0, 0, 0, 1])
        )
    )
    stand_center_to_root_frame: dict[str, rcs.common.Pose] = field(
        default_factory=lambda: rcs.common.Pose(
            translation=np.array([0.4, 0, 0.02]), quaternion=np.array([0, 0, 0.707, 0.707])
        )
    )
    mat_center_to_root_frame: dict[str, rcs.common.Pose] = field(
        default_factory=lambda: rcs.common.Pose(
            translation=np.array([0.5, 0.3, 0.02]), quaternion=np.array([0, 0, 0, 1])
        )
    )

    ring_site = "ring_site"
    mat_site = "mat_site"
    mat_body = "mat_body"


class TransferGateTask(Task[TransferGateTaskConfig]):
    @staticmethod
    def add_task_mujoco(cfg: TransferGateTaskConfig, composer: ModelComposer, env_cfg: SimEnvCreatorConfig):
        """Add task-specific elements to the Mujoco scene."""
        object2world = cfg.object_center_to_root_frame * env_cfg.root_frame_to_world
        for xml in cfg.objects_xml.values():
            composer.add_object_world_frame(
                xml, object_prefix=cfg.prefix, pose=object2world, register_root_relative_replay_free_joints=True
            )

        stand2world = cfg.stand_center_to_root_frame * env_cfg.root_frame_to_world
        for xml in cfg.stand_xml.values():
            composer.add_object_world_frame(
                xml, object_prefix=cfg.prefix, pose=stand2world, register_root_relative_replay_free_joints=True
            )

        mat2world = cfg.mat_center_to_root_frame * env_cfg.root_frame_to_world
        for xml in cfg.mat_xml.values():
            composer.add_object_world_frame(
                xml, object_prefix=cfg.prefix, pose=mat2world, register_root_relative_replay_free_joints=True
            )

    @staticmethod
    def add_task_env(
        cfg: TransferGateTaskConfig, env: gym.Env, simulation: Sim, env_cfg: SimEnvCreatorConfig
    ) -> gym.Env:
        """Add task-specific wrappers to the environment."""
        _ = simulation
        object2world = cfg.object_center_to_root_frame * env_cfg.root_frame_to_world
        obj_joint_names = [cfg.prefix + joint for joint in cfg.objects_joints.values()]

        # For positioning target objects
        env = RandomSquareObjsPos(
            env,
            x_width=cfg.obj_x_width,
            y_width=cfg.obj_y_width,
            z_init=cfg.z_init,
            center2world=object2world,
            include_rotation=cfg.include_rotation,
            obj_joint_names=obj_joint_names,
            obj_position_margin=cfg.obj_position_margin,
        )

        return TaskStageWrapper(
            env,
            TransferGateStage(
                target_body=next(iter(cfg.objects_xml)),
                ring_site=cfg.ring_site,
                mat_body=cfg.mat_body,
                mat_dimensions=np.array([0.12, 0.12]),
                prefix=cfg.prefix,
            ),
        )


rcs.TASKS["transfer_gate"] = TransferGateTask


class TransferGateEnvConfig(VentionSceneFR3Duo):

    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.task_cfg = TransferGateTaskConfig()
        return cfg


gym.register(id="duobench/transfer_gate", entry_point=TransferGateEnvConfig())
