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


@dataclass(kw_only=True)
class JoinBlocksTaskConfig(BaseTaskConfig):

    objects_to_add: list[str] = field(default_factory=lambda: ["h_block", "p_block", "wall"])
    include_rotation: bool = False

    task_instructions = "join the two blocks using the peg on the left block and join the free socket of the right block with the peg on the wall"
    hard_reset = False
    task_id: str = "join_blocks"
    x_width = 0.1
    y_width = 0.15
    z_init = 0.02
    obj_position_margin = 0.08
    object_joint_names: list[str] = field(default_factory=lambda: ["p_block_pblock_joint", "h_block_hblock_joint"])
    object2root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(
            translation=np.array([0.7, 0.0, 0.05]), quaternion=np.array([0, 0, 0, 1])
        )
    )
    object_center_to_root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(
            # 0.02 = z_init
            translation=np.array([0.7, 0.0, 0.02]),
            quaternion=np.array([0, 0, 0, 1]),
        )
    )

    h_block = "h_block_hblock"
    marker_block = "h_block_marker_block"
    marker_wall = "h_block_marker_wall"
    h_block_joint = "h_block_hblock_joint"

    p_block = "p_block_pblock"
    block_peg_name = "p_block_block_peg"
    p_block_joint = "p_block_pblock_joint"

    wall = "wall_wall"
    wall_peg_name = "wall_wall_peg"
    block_dist = 0.01
    wall_dist = 0.01


class JoinBlocksTask(Task[JoinBlocksTaskConfig]):

    @staticmethod
    def add_task_mujoco(cfg: JoinBlocksTaskConfig, composer: ModelComposer, env_cfg: SimEnvCreatorConfig):
        """Add task-specific elements to the Mujoco scene."""
        object2world = env_cfg.root_frame_to_world * cfg.object_center_to_root_frame

        for name in cfg.objects_to_add:
            path = rcs.OBJECT_PATHS[name]

            composer.add_object_world_frame(
                path, object_prefix=name + "_", pose=object2world, register_root_relative_replay_free_joints=True
            )

    @staticmethod
    def add_task_env(cfg: JoinBlocksTaskConfig, env: gym.Env, simulation: Sim, env_cfg: SimEnvCreatorConfig) -> gym.Env:
        """Add task-specific wrappers to the environment."""
        _ = simulation

        randomise_center_left = rcs.common.Pose(
            translation=np.array([0.5, -0.2, 0.05]), quaternion=np.array([1, 0, 0, 0])
        )
        randomise_center_right = rcs.common.Pose(
            translation=np.array([0.5, 0.2, 0.05]), quaternion=np.array([1, 1, 0, 0])
        )
        object2world_left = env_cfg.root_frame_to_world * randomise_center_left
        object2world_right = env_cfg.root_frame_to_world * randomise_center_right

        env = RandomSquareObjsPos(
            env=env,
            x_width=cfg.x_width,
            y_width=cfg.y_width,
            z_init=cfg.z_init,
            # center2world=object2world,
            center2world=object2world_left,
            include_rotation=cfg.include_rotation,
            obj_joint_names=[cfg.object_joint_names[1]],
            obj_position_margin=cfg.obj_position_margin,
        )

        env = RandomSquareObjsPos(
            env=env,
            x_width=cfg.x_width,
            y_width=cfg.y_width,
            z_init=cfg.z_init,
            # center2world=object2world,
            center2world=object2world_right,
            include_rotation=cfg.include_rotation,
            obj_joint_names=[cfg.object_joint_names[0]],
            obj_position_margin=cfg.obj_position_margin,
        )

        return TaskStageWrapper(env, JoinBlocksStage(cfg))


class JoinBlocksStage(TaskStage):
    def __init__(self, cfg: JoinBlocksTaskConfig):
        super().__init__(
            max_stage=3,
            internal_state={
                "approaching_done": False,
                "blocks_connected": False,
                "wall_connected": False,
                "holding_on_wall": False,
            },
            stage_to_subinstructions={
                0: "approach and grasp both blocks",
                1: "connect the two blocks",
                2: "connect the assembled blocks to the wall",
                3: "task completed; the blocks are connected to the wall and the robot is holding them there",
            },
            instruction=cfg.task_instructions,
        )
        self.cfg = cfg

    def update_internal_state(self, sim: Sim):
        block_peg_pos = get_geom_pos(sim.model, sim.data, self.cfg.block_peg_name)
        wall_peg_pos = get_geom_pos(sim.model, sim.data, self.cfg.wall_peg_name)
        wall_marker_pos = get_geom_pos(sim.model, sim.data, self.cfg.marker_wall)
        block_marker_pos = get_geom_pos(sim.model, sim.data, self.cfg.marker_block)

        h_block_body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, self.cfg.h_block)
        p_block_body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, self.cfg.p_block)

        left_contacts = set(get_bodies_in_contact_with_gripper_pad(sim, "left"))
        right_contacts = set(get_bodies_in_contact_with_gripper_pad(sim, "right"))
        block_body_ids = {h_block_body_id, p_block_body_id}

        approaching_done = bool(left_contacts & block_body_ids) and bool(right_contacts & block_body_ids)

        diff_block = abs(block_peg_pos - block_marker_pos)
        blocks_connected = diff_block[0] < 0.05 and diff_block[1] < 0.05 and diff_block[2] < 0.05

        diff_wall = abs(wall_peg_pos - wall_marker_pos)
        wall_connected = diff_wall[0] < 0.05 and diff_wall[1] < 0.05 and diff_wall[2] < 0.01

        holding_on_wall = (
            wall_connected and bool(left_contacts & block_body_ids) and bool(right_contacts & block_body_ids)
        )

        self.internal_state["approaching_done"] = self.internal_state["approaching_done"] or approaching_done
        self.internal_state["blocks_connected"] = self.internal_state["blocks_connected"] or blocks_connected
        self.internal_state["wall_connected"] = self.internal_state["wall_connected"] or wall_connected
        self.internal_state["holding_on_wall"] = holding_on_wall

        self.update_stage()

    def update_stage(self):
        if self.stage == 0 and self.internal_state["approaching_done"]:
            self.stage = 1
        elif self.stage == 1 and self.internal_state["blocks_connected"]:
            self.stage = 2
        elif self.stage == 2 and self.internal_state["holding_on_wall"]:
            self.stage = 3


rcs.TASKS.update({"join_blocks": JoinBlocksTask})


class JoinBlocksEnvConfig(VentionSceneFR3Duo):

    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.task_cfg = JoinBlocksTaskConfig()
        return cfg


gym.register(id="duobench/join_blocks", entry_point=JoinBlocksEnvConfig())
