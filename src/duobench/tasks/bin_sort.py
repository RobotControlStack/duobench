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


class BinSortStage(TaskStage):
    INSTRUCTION = "use the left arm to place the white cube in the white bowl; use the right arm to place the black cube in the black bowl"

    def __init__(self, goals, prefix):
        super().__init__(
            max_stage=5,
            internal_state={
                "left_box_picked": False,
                "right_box_picked": False,
                "left_box_placed": False,
                "right_box_placed": False,
            },
            stage_to_subinstructions={
                0: "pick up at least one cube",
                1: "pick up the other cube or place the picked cube in the correct bowl",
                2: "place at least one cube in the correct bowl",
                3: "pick up the second cube",
                4: "place the second cube in the correct bowl",
                5: "task completed; both cubes are placed correctly",
            },
            instruction=self.INSTRUCTION,
        )
        self.goals = goals
        self.prefix = prefix

    def update_internal_state(self, sim):
        # goal[0] is left, goal[1] is right
        left_goal_name, right_goal_name = tuple(self.goals)
        left_goal = self.goals[left_goal_name]
        right_goal = self.goals[right_goal_name]
        left_object_body_ids = [
            mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, self.prefix + obj + "_body") for obj in left_goal
        ]
        right_object_body_ids = [
            mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, self.prefix + obj + "_body") for obj in right_goal
        ]

        left_goal_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, self.prefix + left_goal_name + "_body")
        right_goal_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, self.prefix + right_goal_name + "_body")

        left_goal_pos = sim.data.xpos[left_goal_id]
        right_goal_pos = sim.data.xpos[right_goal_id]
        # Check if boxes are placed
        for left_object_body_id in left_object_body_ids:
            left_object_pos = sim.data.xpos[left_object_body_id]
            if (
                np.linalg.norm(left_goal_pos[:2] - left_object_pos[:2]) <= 0.06
                and abs(left_goal_pos[2] - left_object_pos[2]) <= 0.2
            ):
                self.internal_state["left_box_placed"] = True

        for right_object_body_id in right_object_body_ids:
            right_object_pos = sim.data.xpos[right_object_body_id]
            if (
                np.linalg.norm(right_goal_pos[:2] - right_object_pos[:2]) <= 0.06
                and abs(right_goal_pos[2] - right_object_pos[2]) <= 0.2
            ):
                self.internal_state["right_box_placed"] = True

        # Check if box is picked. If placed, then we can consider it as picked as well, since the task is sequential.
        left_box_picked = False
        left_box_picked_bodies = get_bodies_in_contact_with_gripper_pad(sim, "left")
        for left_object_body_id in left_object_body_ids:
            if left_object_body_id in left_box_picked_bodies:
                left_box_picked = True
        self.internal_state["left_box_picked"] = left_box_picked or self.internal_state["left_box_placed"]

        right_box_picked = False
        right_box_picked_bodies = get_bodies_in_contact_with_gripper_pad(sim, "right")
        for right_object_body_id in right_object_body_ids:
            if right_object_body_id in right_box_picked_bodies:
                right_box_picked = True
        self.internal_state["right_box_picked"] = right_box_picked or self.internal_state["right_box_placed"]
        self.update_stage()

    def update_stage(self):
        lbpk = self.internal_state["left_box_picked"]
        rbpk = self.internal_state["right_box_picked"]
        lbpl = self.internal_state["left_box_placed"]
        rbpl = self.internal_state["right_box_placed"]

        one_cube_picked = lbpk or rbpk
        both_cubes_picked = lbpk and rbpk
        one_cube_placed = lbpl or rbpl
        both_cubes_placed = lbpl and rbpl
        second_cube_picked_after_first_placed = (lbpl and rbpk and not rbpl) or (rbpl and lbpk and not lbpl)

        if both_cubes_placed:
            self.stage = self.max_stage
        elif self.stage >= 3 and second_cube_picked_after_first_placed:
            self.stage = 4
        elif one_cube_placed:
            self.stage = 3
        elif both_cubes_picked:
            self.stage = 2
        elif one_cube_picked:
            self.stage = 1
        else:
            self.stage = 0


@dataclass(kw_only=True)
class BinSortTaskConfig(BaseTaskConfig):

    z_init = 0.02
    task_id: str = "bin_sort"
    bowl_radius = 0.08
    spawn_clearance = 0.1

    x_width = 0.35
    y_width = 0.25
    obj_position_margin = 0.08
    prefix = "BinSortTask_"
    include_rotation = True
    objects_xml: dict[str, str] = field(
        default_factory=lambda: {
            "black_box": rcs.OBJECT_PATHS["parallel_pick_black_box"],
            # "black_hex_cylinder": rcs.OBJECT_PATHS["parallel_pick_black_hex_cylinder"],
            # "black_oct_cylinder": rcs.OBJECT_PATHS["parallel_pick_black_oct_cylinder"],
            # "black_dec_cylinder": rcs.OBJECT_PATHS["parallel_pick_black_dec_cylinder"],
            "white_box": rcs.OBJECT_PATHS["parallel_pick_white_box"],
            # "white_hex_cylinder": rcs.OBJECT_PATHS["parallel_pick_white_hex_cylinder"],
            # "white_oct_cylinder": rcs.OBJECT_PATHS["parallel_pick_white_oct_cylinder"],
            # "white_dec_cylinder": rcs.OBJECT_PATHS["parallel_pick_white_dec_cylinder"],
        }
    )

    task_objects: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "left": {
                "black_box": "black_box_joint",
                # "black_hex_cylinder": "black_hex_cylinder_joint",
                # "black_oct_cylinder": "black_oct_cylinder_joint",
                # "black_dec_cylinder": "black_dec_cylinder_joint",
            },
            "right": {
                "white_box": "white_box_joint",
                # "white_hex_cylinder": "white_hex_cylinder_joint",
                # "white_oct_cylinder": "white_oct_cylinder_joint",
                # "white_dec_cylinder": "white_dec_cylinder_joint",
            },
        }
    )

    object_center_to_root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(
            translation=np.array([0.5, -0.1, 0.02]), quaternion=np.array([0, 0, 0, 1])
        )
    )
    objects_center_to_root_frame: dict[str, rcs.common.Pose] = field(
        default_factory=lambda: {
            # 0.02 = z_init
            "left": rcs.common.Pose(translation=np.array([0.5, -0.2, 0.02]), quaternion=np.array([0, 0, 0, 1])),
            "right": rcs.common.Pose(translation=np.array([0.5, 0.2, 0.02]), quaternion=np.array([0, 0, 0, 1])),
        }
    )

    goals_xml: dict[str, str] = field(
        default_factory=lambda: {
            "white_bowl": rcs.OBJECT_PATHS["parallel_pick_white_bowl"],
            "black_bowl": rcs.OBJECT_PATHS["parallel_pick_black_bowl"],
        }
    )
    goals_objects: dict[str, list[str]] = field(
        default_factory=lambda: {
            "white_bowl": [
                "white_box",
                # "white_hex_cylinder",
                # "white_oct_cylinder",
                # "white_dec_cylinder",
            ],
            "black_bowl": [
                "black_box",
                # "black_hex_cylinder",
                # "black_oct_cylinder",
                # "black_dec_cylinder",
            ],
        }
    )

    goals_center_to_root_frame: dict[str, rcs.common.Pose] = field(
        default_factory=lambda: {
            # 0.02 = z_init
            "white_bowl": rcs.common.Pose(translation=np.array([0.6, 0.4, 0.02]), quaternion=np.array([0, 0, 0, 1])),
            "black_bowl": rcs.common.Pose(translation=np.array([0.6, -0.4, 0.02]), quaternion=np.array([0, 0, 0, 1])),
        }
    )


class BinSortTask(Task[BinSortTaskConfig]):
    @staticmethod
    def add_task_mujoco(cfg: BinSortTaskConfig, composer: ModelComposer, env_cfg: SimEnvCreatorConfig):
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
    def add_task_env(cfg: BinSortTaskConfig, env: gym.Env, simulation: Sim, env_cfg: SimEnvCreatorConfig) -> gym.Env:
        """Add task-specific wrappers to the environment."""
        _ = simulation
        # For positioning target objects

        for k, v in cfg.objects_center_to_root_frame.items():
            joints = cfg.task_objects[k]
            object2world = v * env_cfg.root_frame_to_world
            obj_joint_names = [cfg.prefix + joint + "_joint" for joint in joints]
            target_bowl_name = "black_bowl" if k == "left" else "white_bowl"
            env = RandomSquareObjsPos(
                env,
                x_width=cfg.x_width,
                y_width=cfg.y_width,
                z_init=cfg.z_init,
                center2world=object2world,
                include_rotation=cfg.include_rotation,
                obj_joint_names=obj_joint_names,
                obj_position_margin=cfg.obj_position_margin,
                avoid_joint_names=[cfg.prefix + target_bowl_name + "_joint"],
                avoid_position_margin=cfg.spawn_clearance,
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
        return TaskStageWrapper(env, BinSortStage(cfg.goals_objects, cfg.prefix))


rcs.TASKS["bin_sort"] = BinSortTask


class BinSortEnvConfig(VentionSceneFR3Duo):

    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.task_cfg = BinSortTaskConfig()
        return cfg


gym.register(id="duobench/bin_sort", entry_point=BinSortEnvConfig())
