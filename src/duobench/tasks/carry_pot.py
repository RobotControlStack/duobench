from dataclasses import dataclass, field

import gymnasium as gym
import mujoco as mj
import numpy as np
import rcs
from rcs.envs.scenes import BaseTaskConfig, SimEnvCreatorConfig, Task
from rcs.sim.composer import ModelComposer
from rcs.sim.sim import Sim

from duobench.tasks import TaskStage, TaskStageWrapper
from duobench.utils.helper_wrappers import RandomSquareObjsPos, body_pose_in_site_frame
from duobench.utils.vention_config import VentionSceneFR3Duo


class CarryPotStage(TaskStage):
    INSTRUCTION = "use two arms to carry the pot at the handle on the stove"

    def __init__(self, target_body: str, stove_site: str, stove_dimensions: np.ndarray, prefix: str = "CarryPotTask_"):
        super().__init__(
            max_stage=3,
            internal_state={
                "one_arm_handle_contact": False,
                "both_arms_handle_contact": False,
                "pot_lifted": False,
                "pot_on_stove": False,
            },
            stage_to_subinstructions={
                0: "make contact with a pot handle",
                1: "grasp the pot by both handles and lift it",
                2: "place the pot onto the stove",
                3: "task completed; the pot is on the stove",
            },
            instruction=self.INSTRUCTION,
        )
        self.target_body = target_body
        self.stove_site = stove_site
        self.stove_dimensions = stove_dimensions
        self.prefix = prefix
        self.initial_pot_height: float | None = None
        self.handle_mesh_names = (
            f"{prefix}handle_left_0",
            f"{prefix}handle_left_1",
            f"{prefix}handle_left_2",
            f"{prefix}handle_left_3",
            f"{prefix}handle_right_0",
            f"{prefix}handle_right_1",
            f"{prefix}handle_right_2",
            f"{prefix}handle_right_3",
        )
        self.left_gripper_pad_geom_names = (
            "gripperleft_left_pad1",
            "gripperleft_left_pad2",
            "gripperleft_right_pad1",
            "gripperleft_right_pad2",
        )
        self.right_gripper_pad_geom_names = (
            "gripperright_left_pad1",
            "gripperright_left_pad2",
            "gripperright_right_pad1",
            "gripperright_right_pad2",
        )

    def reset(self):
        super().reset()
        self.initial_pot_height = None

    def _geom_ids_from_names(self, sim: Sim, obj_type: mj.mjtObj, names: tuple[str, ...]) -> set[int]:
        ids: set[int] = set()
        for name in names:
            obj_id = mj.mj_name2id(sim.model, obj_type, name)
            if obj_id != -1:
                ids.add(obj_id)
        return ids

    def _handle_geom_ids(self, sim: Sim) -> set[int]:
        mesh_ids = self._geom_ids_from_names(sim, mj.mjtObj.mjOBJ_MESH, self.handle_mesh_names)
        handle_geom_ids: set[int] = set()
        for geom_id in range(sim.model.ngeom):
            if sim.model.geom_dataid[geom_id] in mesh_ids:
                handle_geom_ids.add(geom_id)
        return handle_geom_ids

    def _arm_has_handle_contact(self, sim: Sim, gripper_pad_geom_ids: set[int], handle_geom_ids: set[int]) -> bool:
        return any(
            (contact.geom1 in gripper_pad_geom_ids and contact.geom2 in handle_geom_ids)
            or (contact.geom2 in gripper_pad_geom_ids and contact.geom1 in handle_geom_ids)
            for contact in sim.data.contact
        )

    def update_internal_state(self, sim: Sim):
        stove_site_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_SITE, self.prefix + self.stove_site)
        target_body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, self.prefix + self.target_body + "_body")
        assert stove_site_id != -1, f"stove site {self.prefix + self.stove_site} not found in the Mujoco model."
        assert target_body_id != -1, f"Target body {self.prefix + self.target_body} not found in the Mujoco model."

        if self.initial_pot_height is None:
            self.initial_pot_height = float(sim.data.xpos[target_body_id][2])

        handle_geom_ids = self._handle_geom_ids(sim)
        left_pad_geom_ids = self._geom_ids_from_names(sim, mj.mjtObj.mjOBJ_GEOM, self.left_gripper_pad_geom_names)
        right_pad_geom_ids = self._geom_ids_from_names(sim, mj.mjtObj.mjOBJ_GEOM, self.right_gripper_pad_geom_names)

        left_arm_handle_contact = self._arm_has_handle_contact(sim, left_pad_geom_ids, handle_geom_ids)
        right_arm_handle_contact = self._arm_has_handle_contact(sim, right_pad_geom_ids, handle_geom_ids)
        one_arm_handle_contact = left_arm_handle_contact or right_arm_handle_contact
        both_arms_handle_contact = left_arm_handle_contact and right_arm_handle_contact

        current_pot_height = float(sim.data.xpos[target_body_id][2])
        pot_lifted = (
            both_arms_handle_contact
            and self.initial_pot_height is not None
            and current_pot_height > self.initial_pot_height + 0.03
        )

        stove_target_body_T = body_pose_in_site_frame(sim.data, target_body_id, stove_site_id)
        stove_target_body_pos = stove_target_body_T[:3, 3]
        target_within_stove_x = abs(stove_target_body_pos[0]) <= self.stove_dimensions[0]
        target_within_stove_y = abs(stove_target_body_pos[1]) <= self.stove_dimensions[1]
        target_within_stove_z = abs(stove_target_body_pos[2]) <= 0.05
        pot_on_stove_now = target_within_stove_x and target_within_stove_y and target_within_stove_z

        self.internal_state["one_arm_handle_contact"] = (
            self.internal_state["one_arm_handle_contact"] or one_arm_handle_contact
        )
        self.internal_state["both_arms_handle_contact"] = (
            self.internal_state["both_arms_handle_contact"] or both_arms_handle_contact
        )
        self.internal_state["pot_lifted"] = self.internal_state["pot_lifted"] or pot_lifted
        self.internal_state["pot_on_stove"] = self.internal_state["pot_on_stove"] or pot_on_stove_now

        self.update_stage()

    def update_stage(self):
        if self.internal_state["pot_on_stove"] and self.internal_state["pot_lifted"]:
            self.stage = 3
        elif self.internal_state["pot_lifted"]:
            self.stage = 2
        elif self.internal_state["one_arm_handle_contact"]:
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
class CarryPotTaskConfig(BaseTaskConfig):

    z_init = 0.02
    task_id: str = "carry_pot"
    stove_x_width = 0.08
    stove_y_width = 0.08

    obj_x_width = 0.2
    obj_y_width = 0.2

    obj_position_margin = 0.08
    prefix = "CarryPotTask_"
    include_rotation = False

    objects_xml: dict[str, str] = field(
        default_factory=lambda: {
            "pot": rcs.OBJECT_PATHS["carry_pot_pot"],
        }
    )
    objects_joints: dict[str, str] = field(
        default_factory=lambda: {
            "pot": "pot_joint",
        }
    )

    stove_xml: dict[str, str] = field(
        default_factory=lambda: {
            "stove": rcs.OBJECT_PATHS["carry_pot_stove"],
        }
    )

    object_center_to_root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(
            translation=np.array([0.5, -0.1, 0.02]), rpy_vector=np.array([0, 0, np.pi / 2])
        )
    )
    stove_center_to_root_frame: dict[str, rcs.common.Pose] = field(
        default_factory=lambda: rcs.common.Pose(
            translation=np.array([0.5, 0.3, 0.02]), quaternion=np.array([0, 0, 0, 1])
        )
    )

    stove_site = "stove_site"


class CarryPotTask(Task[CarryPotTaskConfig]):
    @staticmethod
    def add_task_mujoco(cfg: CarryPotTaskConfig, composer: ModelComposer, env_cfg: SimEnvCreatorConfig):
        """Add task-specific elements to the Mujoco scene."""
        object2world = cfg.object_center_to_root_frame * env_cfg.root_frame_to_world
        for xml in cfg.objects_xml.values():
            composer.add_object_world_frame(
                xml, object_prefix=cfg.prefix, pose=object2world, register_root_relative_replay_free_joints=True
            )

        stove2world = cfg.stove_center_to_root_frame * env_cfg.root_frame_to_world
        for xml in cfg.stove_xml.values():
            composer.add_object_world_frame(
                xml, object_prefix=cfg.prefix, pose=stove2world, register_root_relative_replay_free_joints=True
            )

    @staticmethod
    def add_task_env(cfg: CarryPotTaskConfig, env: gym.Env, simulation: Sim, env_cfg: SimEnvCreatorConfig) -> gym.Env:
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
            CarryPotStage(
                target_body=next(iter(cfg.objects_xml)),
                stove_site=cfg.stove_site,
                stove_dimensions=np.array([0.12, 0.12]),
                prefix=cfg.prefix,
            ),
        )


rcs.TASKS["carry_pot"] = CarryPotTask


class CarryPotEnvConfig(VentionSceneFR3Duo):

    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.task_cfg = CarryPotTaskConfig()
        return cfg


gym.register(id="duobench/carry_pot", entry_point=CarryPotEnvConfig())
