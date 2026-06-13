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
)
from duobench.utils.vention_config import VentionSceneFR3Duo


class BlockBalanceStage(TaskStage):
    def __init__(self, cfg: "BlockBalanceTaskConfig", prefix: str = "Block_stacking_"):
        super().__init__(
            max_stage=5,
            internal_state={
                "beam_grasped": False,
                "beam_on_cube": False,
                "both_cubes_grasped": False,
                "both_cubes_on_beam": False,
                "hands_retracted": False,
                "beam_balanced": False,
            },
            stage_to_subinstructions={
                0: "grasp the beam with one arm",
                1: "place the beam on the small cube",
                2: "grasp both rectangles",
                3: "place both rectangles onto the beam",
                4: "release and retract while keeping the beam balanced",
                5: "task completed; the beam remains balanced after release",
            },
            instruction=cfg.task_instructions,
        )
        self.prefix = prefix
        self.cfg = cfg
        self.stable_steps = 0

        self.beam_body_name = self.prefix + "pink_beam"
        self.red_cube_body_name = self.prefix + "big_red_cube"
        self.blue_rectangle_body_name = self.prefix + "blue_rectangle"
        self.green_rectangle_body_name = self.prefix + "green_rectangle"

        self.beam_geom_name = self.prefix + "pink_beam_geom"
        self.red_cube_geom_name = self.prefix + "big_red_cube_geom"
        self.blue_rectangle_geom_name = self.prefix + "blue_rectangle_geom"
        self.green_rectangle_geom_name = self.prefix + "green_rectangle_geom"

        self.required_stable_steps = 5
        self.balance_angle_tolerance_rad = np.deg2rad(12.0)
        self.linear_velocity_tolerance = 0.08
        self.angular_velocity_tolerance = 0.8

    def reset(self):
        super().reset()
        self.stable_steps = 0

    def _body_in_contact_pairs(self, sim: Sim) -> set[tuple[str, str]]:
        contacts: set[tuple[str, str]] = set()
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            g1 = sim.model.geom(contact.geom1).name
            g2 = sim.model.geom(contact.geom2).name
            if g1 is not None and g2 is not None:
                contacts.add((g1, g2))
                contacts.add((g2, g1))
        return contacts

    def _body_has_any_contact(self, sim: Sim, body_id: int) -> bool:
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            geom1_body_id = sim.model.geom_bodyid[contact.geom1]
            geom2_body_id = sim.model.geom_bodyid[contact.geom2]
            if body_id in (geom1_body_id, geom2_body_id):
                return True
        return False

    def update_internal_state(self, sim: Sim):
        beam_body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, self.beam_body_name)
        blue_rectangle_body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, self.blue_rectangle_body_name)
        green_rectangle_body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, self.green_rectangle_body_name)
        red_cube_body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, self.red_cube_body_name)

        contact_pairs = self._body_in_contact_pairs(sim)
        left_arm_contacts = set(get_bodies_in_contact_with_gripper_pad(sim, "left"))
        right_arm_contacts = set(get_bodies_in_contact_with_gripper_pad(sim, "right"))

        beam_grasped = beam_body_id in left_arm_contacts or beam_body_id in right_arm_contacts

        beam_on_cube = (self.red_cube_geom_name, self.beam_geom_name) in contact_pairs or self.internal_state[
            "beam_on_cube"
        ]

        cubes_contacted_by_left = {
            body_id for body_id in (blue_rectangle_body_id, green_rectangle_body_id) if body_id in left_arm_contacts
        }
        cubes_contacted_by_right = {
            body_id for body_id in (blue_rectangle_body_id, green_rectangle_body_id) if body_id in right_arm_contacts
        }
        both_cubes_grasped = (
            len(cubes_contacted_by_left | cubes_contacted_by_right) == 2
            and bool(cubes_contacted_by_left)
            and bool(cubes_contacted_by_right)
        )

        both_cubes_on_beam = (
            (self.blue_rectangle_geom_name, self.beam_geom_name) in contact_pairs
            and (self.green_rectangle_geom_name, self.beam_geom_name) in contact_pairs
            and beam_on_cube
        )

        hands_retracted = not (
            left_arm_contacts & {beam_body_id, blue_rectangle_body_id, green_rectangle_body_id, red_cube_body_id}
            or right_arm_contacts & {beam_body_id, blue_rectangle_body_id, green_rectangle_body_id, red_cube_body_id}
        )

        beam_qvel_adr = sim.model.jnt_dofadr[
            mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_JOINT, self.prefix + "pink_beam_joint")
        ]
        beam_linear_velocity = np.linalg.norm(sim.data.qvel[beam_qvel_adr : beam_qvel_adr + 3])
        beam_angular_velocity = np.linalg.norm(sim.data.qvel[beam_qvel_adr + 3 : beam_qvel_adr + 6])
        beam_rotation = sim.data.xmat[beam_body_id].reshape(3, 3)
        beam_up = beam_rotation[:, 2]
        vertical_alignment = np.clip(np.dot(beam_up, np.array([0.0, 0.0, 1.0])), -1.0, 1.0)
        beam_tilt = np.arccos(vertical_alignment)

        beam_balanced_now = (
            both_cubes_on_beam
            and beam_on_cube
            and hands_retracted
            and beam_tilt <= self.balance_angle_tolerance_rad
            and beam_linear_velocity <= self.linear_velocity_tolerance
            and beam_angular_velocity <= self.angular_velocity_tolerance
            and self._body_has_any_contact(sim, beam_body_id)
            and self._body_has_any_contact(sim, blue_rectangle_body_id)
            and self._body_has_any_contact(sim, green_rectangle_body_id)
        )

        if beam_balanced_now:
            self.stable_steps += 1
        else:
            self.stable_steps = 0

        self.internal_state["beam_grasped"] = beam_grasped
        self.internal_state["beam_on_cube"] = beam_on_cube
        self.internal_state["both_cubes_grasped"] = both_cubes_grasped
        self.internal_state["both_cubes_on_beam"] = both_cubes_on_beam
        self.internal_state["hands_retracted"] = hands_retracted
        self.internal_state["beam_balanced"] = self.stable_steps >= self.required_stable_steps

        self.update_stage()

    def update_stage(self):
        if self.internal_state["beam_balanced"]:
            new_stage = 5
        elif self.internal_state["both_cubes_on_beam"]:
            new_stage = 4
        elif self.internal_state["both_cubes_grasped"]:
            new_stage = 3
        elif self.internal_state["beam_on_cube"]:
            new_stage = 2
        elif self.internal_state["beam_grasped"]:
            new_stage = 1
        else:
            new_stage = 0

        self.stage = max(self.stage, new_stage)


@dataclass(kw_only=True)
class BlockBalanceTaskConfig(BaseTaskConfig):

    z_init = 0.02
    task_instructions = "place the beam on the cube and then place the other blocks on the beam simultaneously using one arm for each cube"
    task_id: str = "block_balance"
    bowl_radius = 0.06
    table_hight = 0.2

    x_width = 0.35
    y_width = 0.35
    obj_position_margin = 0.08
    prefix = "BlockBalanceTask_"
    include_rotation = False
    objects_xml: dict[str, str] = field(
        default_factory=lambda: {
            "block_stacking_big_red_cube": rcs.OBJECT_PATHS["block_stacking_big_red_cube"],
            "block_stacking_blue_rectangle": rcs.OBJECT_PATHS["block_stacking_blue_rectangle"],
            "block_stacking_green_rectangle": rcs.OBJECT_PATHS["block_stacking_green_rectangle"],
            "block_stacking_pink_beam": rcs.OBJECT_PATHS["block_stacking_pink_beam"],
        }
    )
    objects_joints: dict[str, str] = field(
        default_factory=lambda: {
            "pink_beam": "pink_beam_joint",
            "green_rectangle": "green_rectangle_joint",
            "blue_rectangle": "blue_rectangle_joint",
            "big_red_cube": "big_red_cube_joint",
        }
    )
    object_center_to_root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(
            # 0.02 = z_init
            translation=np.array([0.5, 0.0, 0.02]),
            quaternion=np.array([0, 0, 0, 1]),
        )
    )


class BlockBalanceTask(Task[BlockBalanceTaskConfig]):
    @staticmethod
    def add_task_mujoco(cfg: BlockBalanceTaskConfig, composer: ModelComposer, env_cfg: SimEnvCreatorConfig):
        """Add task-specific elements to the Mujoco scene."""
        object2world = cfg.object_center_to_root_frame * env_cfg.root_frame_to_world

        for xml in cfg.objects_xml.values():
            composer.add_object_world_frame(
                xml, object_prefix=cfg.prefix, pose=object2world, register_root_relative_replay_free_joints=True
            )

    @staticmethod
    def add_task_env(
        cfg: BlockBalanceTaskConfig, env: gym.Env, simulation: Sim, env_cfg: SimEnvCreatorConfig
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
        )

        return TaskStageWrapper(env, BlockBalanceStage(cfg, prefix=cfg.prefix))


rcs.TASKS.update(
    {
        "block_balance": BlockBalanceTask,
    }
)


class BlockBalanceEnvConfig(VentionSceneFR3Duo):

    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.task_cfg = BlockBalanceTaskConfig()
        return cfg


gym.register(id="duobench/block_balance", entry_point=BlockBalanceEnvConfig())
