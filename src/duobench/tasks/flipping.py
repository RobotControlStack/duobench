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
from duobench.utils.helper_wrappers import RandomSquareObjsPos
from duobench.utils.single_arm import SingleArm


class FlippingStage(TaskStage):
    """
    Stages:
      0 - initial (cube resting upright)
      1 - the cube has been touched by the gripper for the first time
      2 - success: the face that was initially up now rests on a side (~90 deg flip)

    The task is considered *unsuccessful* if instead the initially-up face ends up
    pointing down (~180 deg): the ``failed`` flag is set and stage 2 is never reached.
    """

    def __init__(self, cfg: "FlippingTaskConfig", prefix: str = "Flipping_"):
        super().__init__(
            max_stage=2,
            internal_state={"touched": False, "flipped": False, "failed": False},
            stage_to_subinstructions={
                0: "flip the red cube so that its top face ends up on a side",
                1: "the cube has been touched; tip it onto its side",
                2: "task completed; the initially-up face now rests on a side",
            },
            instruction=cfg.task_instructions,
        )
        self.cfg = cfg
        self.prefix = prefix
        # unit direction (in the cube's local frame) of the face that points up at reset
        self.up_local: np.ndarray | None = None

    def reset(self):
        super().reset()
        self.up_local = None

    def _cube_body_id(self, sim: Sim) -> int:
        name = self.prefix + self.cfg.cube_body
        bid = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            msg = f"Body '{name}' not found"
            raise ValueError(msg)
        return bid

    def _touching_gripper(self, sim: Sim, cube_body_id: int) -> bool:
        model, data = sim.model, sim.data
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = int(model.geom_bodyid[c.geom1])
            b2 = int(model.geom_bodyid[c.geom2])
            if cube_body_id not in (b1, b2):
                continue
            other = b2 if b1 == cube_body_id else b1
            other_name = model.body(other).name
            if other_name.startswith("gripper") or "pad" in other_name or "finger" in other_name:
                return True
        return False

    def update_internal_state(self, sim: Sim):
        bid = self._cube_body_id(sim)
        rot = sim.data.xmat[bid].reshape(3, 3)  # world <- cube-local rotation
        world_up = np.array([0.0, 0.0, 1.0])

        # Record which physical face is up at the start (expressed in cube-local coords),
        # so the check is robust to any initial yaw/placement.
        if self.up_local is None:
            self.up_local = rot.T @ world_up

        # z-component of that face normal in the world now: +1 up, 0 on a side, -1 down
        up_z = float(world_up @ (rot @ self.up_local))
        # only classify a resting cube, to avoid latching mid-flip transients
        settled = float(np.linalg.norm(sim.data.cvel[bid])) < self.cfg.settle_speed

        if self._touching_gripper(sim, bid):
            self.internal_state["touched"] = True

        if not self.internal_state["flipped"]:
            if settled and abs(up_z) < self.cfg.side_cos_thresh:
                self.internal_state["flipped"] = True
            elif settled and up_z < -self.cfg.down_cos_thresh:
                self.internal_state["failed"] = True

        self.update_stage()

    def update_stage(self):
        if self.internal_state["flipped"]:
            self.stage = 2
        elif self.internal_state["touched"]:
            self.stage = 1
        else:
            self.stage = 0

    @property
    def info(self) -> dict[str, Any]:
        base = super().info
        base.update(
            {
                "touched": self.internal_state["touched"],
                "flipped": self.internal_state["flipped"],
                "failed": self.internal_state["failed"],
            }
        )
        return base


BLOCK_NAME = "red_cube"

@dataclass(kw_only=True)
class FlippingTaskConfig(BaseTaskConfig):
    task_id: str = "flipping"
    prefix: str = "Flipping_"
    task_instructions: str = "flip the red cube so that the face that is initially on top ends up on a side"

    # red_cube.xml element names (prefixed at attach time)
    cube_body: str = f"{BLOCK_NAME}_body"
    cube_geom: str = f"{BLOCK_NAME}_geom"
    cube_joint: str = f"{BLOCK_NAME}_joint"

    # classification thresholds (on the up-face normal's world z-component)
    side_cos_thresh: float = 0.5  # |z| < 0.5  -> on a side  (angle within [60, 120] deg)
    down_cos_thresh: float = 0.5  #  z < -0.5  -> upside down (angle > 120 deg)
    settle_speed: float = 0.05  # cube spatial-velocity norm below this counts as "at rest"

    objects_xml: dict[str, str] = field(
        default_factory=lambda: {"red_cube": rcs.OBJECT_PATHS[f"{BLOCK_NAME}"]}
    )
    # randomization origin
    object_center_to_root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(translation=np.array([0.5, 0.0, 0.05]))
    )
    rand_x_width: float = 0.2
    rand_y_width: float = 0.4
    obj_position_margin: float = 0.05


class FlippingTask(Task[FlippingTaskConfig]):
    @staticmethod
    def add_task_mujoco(cfg: FlippingTaskConfig, composer: ModelComposer, env_cfg: SimEnvCreatorConfig):
        """Add the red cube to the scene."""
        object2world = env_cfg.root_frame_to_world * cfg.object_center_to_root_frame
        for xml in cfg.objects_xml.values():
            composer.add_object_world_frame(xml, object_prefix=cfg.prefix, pose=object2world)

    @staticmethod
    def add_task_env(
        cfg: FlippingTaskConfig, env: gym.Env, simulation: Sim, env_cfg: SimEnvCreatorConfig
    ) -> gym.Env:
        _ = simulation
        env = RandomSquareObjsPos(
            env,
            center2world=env_cfg.root_frame_to_world * cfg.object_center_to_root_frame,
            obj_joint_names=[f"{cfg.prefix}{cfg.cube_joint}"],
            x_width=cfg.rand_x_width,
            y_width=cfg.rand_y_width,
            z_init=0.0,
            include_rotation=False,
            obj_position_margin=cfg.obj_position_margin,
        )
        return TaskStageWrapper(env, FlippingStage(cfg, prefix=cfg.prefix))


rcs.TASKS.update({"flipping": FlippingTask})


class FlippingEnvConfig(SingleArm):
    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.task_cfg = FlippingTaskConfig()
        return cfg


gym.register(id="duobench/flipping", entry_point=FlippingEnvConfig())
