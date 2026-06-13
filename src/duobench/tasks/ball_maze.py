from dataclasses import dataclass, field
from typing import Any

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


@dataclass(kw_only=True)
class BallMazeTaskConfig(BaseTaskConfig):

    task_id: str = "ball_maze"

    board_number = 2
    object_body = "board"
    include_rotation: bool = True
    task_instructions = "pick up the board and tilt it so the ball roles onto the red square"
    hard_reset = False
    x_width = 0.3
    y_width = 0.3
    z_init = 0.02
    obj_position_margin = 0.08
    object_joint_names: list[str] = field(default_factory=lambda: ["board_board_joint"])
    object2root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(translation=np.array([0.5, 0, 0.05]), quaternion=np.array([0, 0, 0, 1]))
    )
    object_center_to_root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(
            # 0.02 = z_init
            translation=np.array([0.5, 0.0, 0.02]),
            quaternion=np.array([0, 0, 0, 1]),
        )
    )


class BallMazeTask(Task[BallMazeTaskConfig]):

    @staticmethod
    def add_task_mujoco(cfg: BallMazeTaskConfig, composer: ModelComposer, env_cfg: SimEnvCreatorConfig):
        """Add task-specific elements to the Mujoco scene."""
        object2world = env_cfg.root_frame_to_world * cfg.object_center_to_root_frame
        key = "maze{number}"
        board_xml = rcs.OBJECT_PATHS[key.format(number=cfg.board_number)].format(number=cfg.board_number)

        composer.add_object_world_frame(
            board_xml,
            object_prefix=cfg.object_body + "_",
            pose=object2world,
            register_root_relative_replay_free_joints=True,
        )

    @staticmethod
    def add_task_env(cfg: BallMazeTaskConfig, env: gym.Env, simulation: Sim, env_cfg: SimEnvCreatorConfig) -> gym.Env:
        """Add task-specific wrappers to the environment."""
        _ = simulation
        object2world = env_cfg.root_frame_to_world * cfg.object_center_to_root_frame

        env = RandomSquareObjsPos(
            env=env,
            x_width=cfg.x_width,
            y_width=cfg.y_width,
            z_init=cfg.z_init,
            center2world=object2world,
            include_rotation=cfg.include_rotation,
            obj_joint_names=cfg.object_joint_names,
            obj_position_margin=cfg.obj_position_margin,
        )

        return BallMazeTaskWrapper(env, BallMazeStage(cfg))


class BallMazeStage(TaskStage):
    def __init__(self, cfg: BallMazeTaskConfig):
        super().__init__(
            stage=0,
            max_stage=4,
            internal_state={
                "left_arm_contact": False,
                "right_arm_contact": False,
                "both_arms_contact": False,
                "maze_lifted": False,
                "ball_left_start": False,
                "ball_at_goal": False,
                "ball_near_goal": False,
                "maze_on_table": True,
                "hands_released": False,
            },
            stage_to_subinstructions={
                0: "make contact with the maze",
                1: "grasp the maze with both arms",
                2: "lift the maze and move the ball out of the start area",
                3: "guide the ball from the start area toward the goal",
                4: "task completed; the ball has reached the goal",
            },
            instruction=cfg.task_instructions,
        )
        self.cfg = cfg
        self.start_marker = "board_marker_start"
        self.goal_marker = "board_marker_goal"
        self.ball_joint = "board_ball_joint"
        self.ball_name = "board_ball"
        self.board_name = "board_board"
        self.start_dist = 0.02
        self.goal_dist = 0.015
        self.done_goal_radius = 0.10
        self.table_height: float | None = None

    @staticmethod
    def _planar_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))

    def set_table_height(self, table_height: float):
        self.table_height = table_height

    def _board_body_ids(self, sim: Sim) -> set[int]:
        body_ids: set[int] = set()
        for body_id in range(sim.model.nbody):
            body_name = sim.model.body(body_id).name
            if body_name and body_name.startswith("board_") and body_name != self.ball_name:
                body_ids.add(body_id)
        return body_ids

    def _maze_on_table(self, sim: Sim, board_body_ids: set[int], board_height: float) -> bool:
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            geom1_name = sim.model.geom(contact.geom1).name
            geom2_name = sim.model.geom(contact.geom2).name
            body1_id = sim.model.geom_bodyid[contact.geom1]
            body2_id = sim.model.geom_bodyid[contact.geom2]

            if geom1_name == "floor" and body2_id in board_body_ids:
                return True
            if geom2_name == "floor" and body1_id in board_body_ids:
                return True

        if self.table_height is None:
            return False

        return board_height <= self.table_height + 0.02

    def update_internal_state(self, sim: Sim):
        board_body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, self.board_name)
        board_body_ids = self._board_body_ids(sim)
        left_arm_contacts = get_bodies_in_contact_with_gripper_pad(sim, "left")
        right_arm_contacts = get_bodies_in_contact_with_gripper_pad(sim, "right")

        left_arm_contact = any(body_id in board_body_ids for body_id in left_arm_contacts)
        right_arm_contact = any(body_id in board_body_ids for body_id in right_arm_contacts)
        both_arms_contact = left_arm_contact and right_arm_contact
        any_arm_contact = left_arm_contact or right_arm_contact

        board_height = sim.data.xpos[board_body_id][2]
        maze_on_table = self._maze_on_table(sim, board_body_ids, board_height)

        ball_pose = sim.data.body(self.ball_name).xpos
        start_pose = sim.data.body(self.start_marker).xpos
        goal_pose = sim.data.body(self.goal_marker).xpos
        # The ball center is always above the board plane, so using full 3D distance
        # can incorrectly mark "left start" immediately after reset.
        start_dist = self._planar_distance(start_pose, ball_pose)
        goal_dist = np.linalg.norm(goal_pose - ball_pose)

        self.internal_state["left_arm_contact"] = left_arm_contact
        self.internal_state["right_arm_contact"] = right_arm_contact
        self.internal_state["both_arms_contact"] = both_arms_contact
        self.internal_state["maze_lifted"] = both_arms_contact and not maze_on_table
        self.internal_state["ball_left_start"] = bool(
            self.internal_state["ball_left_start"] or start_dist >= self.start_dist
        )
        self.internal_state["ball_at_goal"] = bool(self.internal_state["ball_at_goal"] or goal_dist <= self.goal_dist)
        self.internal_state["ball_near_goal"] = bool(goal_dist <= self.done_goal_radius)
        self.internal_state["maze_on_table"] = maze_on_table
        self.internal_state["hands_released"] = not any_arm_contact

        self.update_stage()

    def update_stage(self):
        if self.internal_state["ball_at_goal"]:
            self.stage = 4
        elif self.internal_state["ball_left_start"]:
            self.stage = 3
        elif self.internal_state["maze_lifted"]:
            self.stage = 2
        elif self.internal_state["left_arm_contact"] or self.internal_state["right_arm_contact"]:
            self.stage = 1
        else:
            self.stage = 0


class BallMazeTaskWrapper(TaskStageWrapper):
    """
    Wrapper to conduct the task-specific soft reset for the ball maze while using the shared task-stage interface.
    """

    def __init__(self, env: gym.Env, stage_tracker: BallMazeStage):
        super().__init__(env, stage_tracker)
        self.stage_tracker: BallMazeStage = stage_tracker
        self.sim = self.env.get_wrapper_attr("sim")
        self.start_marker = stage_tracker.start_marker
        self.ball_joint = stage_tracker.ball_joint

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)

        # The board pose is randomized by inner wrappers during reset, so place the
        # ball only after the full reset chain has finished. Spawn the ball tangent
        # to the maze floor and let the physics settle before the episode starts.
        pos = self.sim.data.body(self.start_marker).xpos.copy()
        ball_radius = 0.01
        spawn_clearance = 0.001
        pos[2] += ball_radius + spawn_clearance

        joint_id = self.sim.model.joint(self.ball_joint).id
        qpos_adr = self.sim.model.jnt_qposadr[joint_id]
        qvel_adr = self.sim.model.jnt_dofadr[joint_id]
        self.sim.data.qpos[qpos_adr : qpos_adr + 3] = pos
        # Reset the free joint velocity so the ball does not inherit motion from the previous rollout.
        self.sim.data.qvel[qvel_adr : qvel_adr + 6] = 0

        mj.mj_forward(self.sim.model, self.sim.data)
        for _ in range(10):
            mj.mj_step(self.sim.model, self.sim.data)
        self.stage_tracker.set_table_height(float(self.sim.data.body(self.stage_tracker.board_name).xpos[2]))
        self.stage_tracker.update_internal_state(self.sim)
        info.update(self.stage_tracker.info)
        return obs, info


rcs.TASKS.update({"ball_maze": BallMazeTask})


class BallMazeEnvConfig(VentionSceneFR3Duo):

    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.task_cfg = BallMazeTaskConfig()
        return cfg


gym.register(id="duobench/ball_maze", entry_point=BallMazeEnvConfig())
