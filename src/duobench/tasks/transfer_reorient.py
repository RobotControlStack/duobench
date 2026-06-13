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
    get_body_euler_xyz,
    get_geom_pos,
)
from duobench.utils.vention_config import VentionSceneFR3Duo


class TransferReorientStage(TaskStage):
    def __init__(
        self,
        cfg: "TransferReorientTaskConfig",
        prefix: str = "TransferReorient_",
    ):
        super().__init__(
            max_stage=4,
            internal_state={
                "grasped_and_lifted": False,
                "both_grippers_contact": False,
                "successfully_transferred": False,
                "inserted": False,
            },
            stage_to_subinstructions={
                0: "pick up the peg",
                1: "bring both grippers into contact with the peg",
                2: "transfer the peg to the other gripper",
                3: "insert the peg into the matching socket",
                4: "task completed; the peg is close to the socket and upright for insertion",
            },
            instruction=cfg.task_instructions,
        )
        self.goal_dist = 0.038
        self.prefix = prefix
        self.cfg = cfg
        self.initial_holder: str | None = None

    def reset(self):
        super().reset()
        self.initial_holder = None

    def update_internal_state(self, sim: Sim):
        marker_pos = get_geom_pos(sim.model, sim.data, self.prefix + self.cfg.marker)
        peg_pose = get_geom_pos(sim.model, sim.data, self.prefix + self.cfg.peg)
        peg_orientation = get_body_euler_xyz(sim.model, sim.data, self.prefix + self.cfg.peg + "_body")
        peg_body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, self.prefix + self.cfg.peg + "_body")
        left_contacts = set(get_bodies_in_contact_with_gripper_pad(sim, "left"))
        right_contacts = set(get_bodies_in_contact_with_gripper_pad(sim, "right"))
        left_gripper_contact = peg_body_id in left_contacts
        right_gripper_contact = peg_body_id in right_contacts

        upright_x = abs(peg_orientation[0]) < 20 or abs(abs(peg_orientation[0]) - 180) < 20
        upright_y = abs(peg_orientation[1]) < 20 or abs(abs(peg_orientation[1]) - 180) < 20
        upright = upright_x and upright_y
        lifted = peg_pose[2] > self.cfg.table_hight + 0.012
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

        socket_dist = float(np.linalg.norm(marker_pos - peg_pose))
        inserted = bool(socket_dist < self.goal_dist and upright)

        self.internal_state["grasped_and_lifted"] = self.internal_state["grasped_and_lifted"] or grasped_and_lifted
        self.internal_state["both_grippers_contact"] = (
            self.internal_state["both_grippers_contact"] or both_grippers_contact
        )
        self.internal_state["successfully_transferred"] = (
            self.internal_state["successfully_transferred"] or successfully_transferred
        )
        self.internal_state["inserted"] = bool(self.internal_state["inserted"] or inserted)

        self.update_stage()

    def update_stage(self):
        if self.internal_state["inserted"]:
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
class TransferReorientTaskConfig(BaseTaskConfig):

    z_init = 0.02
    task_instructions = "grasp the block with the right arm, hand it over to the left arm such that the left arm can easily insert the piece later, then insert the block into the socket with the left arm"
    task_id: str = "transfer_reorient"
    marker = "marker_hex"
    x_width = 0.3
    y_width = 0.3
    obj_position_margin = 0.08
    prefix = "TransferReorient_"
    include_rotation = False
    peg = "peg"
    objects_xml: dict[str, str] = field(
        default_factory=lambda: {
            "transfer_reorient_hex_peg": rcs.OBJECT_PATHS["transfer_reorient_hex_peg"],
            "transfer_reorient_socket": rcs.OBJECT_PATHS["transfer_reorient_socket"],
        }
    )

    objects_joints: dict[str, str] = field(default_factory=lambda: {"peg": "peg_joint"})

    socket_joints: dict[str, str] = field(default_factory=lambda: {"socket": "socket_joint"})
    object_center_to_root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(
            # 0.02 = z_init
            translation=np.array([0.5, -0.2, 0.02]),
            quaternion=np.array([0, 1, 0, 1]),
        )
    )

    socket_center_to_root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(
            translation=np.array([0.5, 0.3, 0.02]), quaternion=np.array([0, 0, 0, 1])
        ),
    )

    table_hight = 1


class TransferReorientTask(Task[TransferReorientTaskConfig]):
    @staticmethod
    def add_task_mujoco(cfg: TransferReorientTaskConfig, composer: ModelComposer, env_cfg: SimEnvCreatorConfig):
        """Add task-specific elements to the Mujoco scene."""
        object2world = cfg.object_center_to_root_frame * env_cfg.root_frame_to_world

        for xml in cfg.objects_xml.values():
            composer.add_object_world_frame(
                xml,
                object_prefix=cfg.prefix,
                pose=object2world,
            )

    @staticmethod
    def add_task_env(
        cfg: TransferReorientTaskConfig, env: gym.Env, simulation: Sim, env_cfg: SimEnvCreatorConfig
    ) -> gym.Env:
        _ = simulation

        object2world = env_cfg.root_frame_to_world * cfg.object_center_to_root_frame
        socket2world = env_cfg.root_frame_to_world * cfg.socket_center_to_root_frame
        obj_joint_names = [cfg.prefix + joint for joint in cfg.objects_joints.values()]
        socket_joint_names = [cfg.prefix + joint for joint in cfg.socket_joints.values()]

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

        # For positioning the Socket
        env = RandomSquareObjsPos(
            env,
            x_width=cfg.x_width,
            y_width=cfg.y_width,
            z_init=cfg.z_init,
            center2world=socket2world,
            include_rotation=cfg.include_rotation,
            obj_joint_names=socket_joint_names,
            obj_position_margin=cfg.obj_position_margin,
        )

        return TaskStageWrapper(env, TransferReorientStage(cfg, prefix=cfg.prefix))


rcs.TASKS.update({"transfer_reorient": TransferReorientTask})


class TransferReorientEnvConfig(VentionSceneFR3Duo):

    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.task_cfg = TransferReorientTaskConfig()
        return cfg


gym.register(id="duobench/transfer_reorient", entry_point=TransferReorientEnvConfig())
