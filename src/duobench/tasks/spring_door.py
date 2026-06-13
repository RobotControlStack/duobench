from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import rcs
from rcs.envs.scenes import BaseTaskConfig, SimEnvCreatorConfig, Task
from rcs.sim.composer import ModelComposer
from rcs.sim.sim import Sim

from duobench.tasks import TaskStageWrapper
from duobench.tasks.hinge_chest import HingeChestStage
from duobench.utils.helper_wrappers import RandomSquareObjsPos
from duobench.utils.vention_config import VentionSceneFR3Duo


class SpringDoorStage(HingeChestStage):
    INSTRUCTION = "use the left arm to open the microwave door, then use the right arm to place the box inside the microwave, and close the door again"

    def __init__(self, goal_site, target_body, door_joint, goal_site_dimensions, prefix):
        super().__init__(goal_site, target_body, door_joint, goal_site_dimensions, prefix)
        self.stage_to_subinstructions = {
            0: "open the microwave OR pick up the box",
            1: "pick up the box AND open the microwave",
            2: "place the box inside the microwave",
            3: "task completed; the box is inside the microwave",
        }


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
class SpringDoorTaskConfig(BaseTaskConfig):

    z_init = 0.02
    task_id: str = "spring_door"
    bowl_radius = 0.06

    x_width = 0.1
    y_width = 0.2
    obj_position_margin = 0.08
    prefix = "SpringDoorTask_"
    include_rotation = True
    objects_xml: dict[str, str] = field(
        default_factory=lambda: {
            "white_box": rcs.OBJECT_PATHS["parallel_pick_white_box"],
        }
    )
    objects_joints: dict[str, str] = field(
        default_factory=lambda: {
            "white_box": "white_box_joint",
        }
    )

    object_center_to_root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(
            # 0.02 = z_init
            translation=np.array([0.4, -0.35, 0.02]),
            quaternion=np.array([0, 0, 0, 1]),
        )
    )
    goals_xml: dict[str, str] = field(
        default_factory=lambda: {"gray_microwave": rcs.OBJECT_PATHS["spring_door_gray_microwave"]}
    )

    goals_center_to_root_frame: dict[str, rcs.common.Pose] = field(
        default_factory=lambda: {
            # 0.02 = z_init
            "gray_microwave": rcs.common.Pose(
                translation=np.array([0.7, 0, 0.01]), quaternion=np.array([0, 0, -0.707, 0.707])
            ),
        }
    )
    goal_site = "placement_site"
    door_joint = "door_joint"
    goal_site_dimensions: np.ndarray = field(default_factory=lambda: np.array([0.08, 0.095, 0.06]))


class SpringDoorTask(Task[SpringDoorTaskConfig]):
    @staticmethod
    def add_task_mujoco(cfg: SpringDoorTaskConfig, composer: ModelComposer, env_cfg: SimEnvCreatorConfig):
        """Add task-specific elements to the Mujoco scene."""
        object2world = cfg.object_center_to_root_frame * env_cfg.root_frame_to_world

        for xml in cfg.objects_xml.values():
            composer.add_object_world_frame(
                xml,
                object_prefix=cfg.prefix,
                pose=object2world,
                register_root_relative_replay_free_joints=True,
            )

        for goal_name, xml in cfg.goals_xml.items():
            center = cfg.goals_center_to_root_frame[goal_name]
            goal2world = center * env_cfg.root_frame_to_world
            composer.add_object_world_frame(
                xml, object_prefix=cfg.prefix, pose=goal2world, register_root_relative_replay_free_joints=True
            )

    @staticmethod
    def add_task_env(cfg: SpringDoorTaskConfig, env: gym.Env, simulation: Sim, env_cfg: SimEnvCreatorConfig) -> gym.Env:
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
        #         obj_joint_names=[cfg.prefix + k + "_joint"],
        #     )
        return TaskStageWrapper(
            env,
            SpringDoorStage(
                goal_site=cfg.goal_site,
                target_body=next(iter(cfg.objects_xml)),
                door_joint=cfg.door_joint,
                goal_site_dimensions=cfg.goal_site_dimensions,
                prefix=cfg.prefix,
            ),
        )


rcs.TASKS["spring_door"] = SpringDoorTask


class SpringDoorEnvConfig(VentionSceneFR3Duo):

    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.task_cfg = SpringDoorTaskConfig()
        return cfg


gym.register(id="duobench/spring_door", entry_point=SpringDoorEnvConfig())
