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


class PickPlaceStage(TaskStage):
    """
    Pick the red cube and place it on top of the green cube.

    Stages (monotonic, each gated on the previous):
      0 - initial
      1 - the red cube has been touched by the gripper
      2 - the red cube has been lifted off the table (while grasped)
      3 - the red cube touches the green cube
      4 - success: red rests on green (stacked, settled) and the gripper is retracted
    """

    def __init__(self, cfg: "PickPlaceTaskConfig", prefix: str = "PickPlace_"):
        super().__init__(
            max_stage=4,
            internal_state={"touched": False, "lifted": False, "touched_green": False, "placed": False},
            stage_to_subinstructions={
                0: "reach for and touch the red cube",
                1: "grasp and lift the red cube off the table",
                2: "move the lifted red cube over the green cube",
                3: "lower the red cube so it rests on the green cube, then let go",
                4: "task completed; the red cube rests on the green cube and the gripper is retracted",
            },
            instruction=cfg.task_instructions,
        )
        self.cfg = cfg
        self.prefix = prefix
        self.red_body = f"{prefix}red_{cfg.cube_body}"
        self.green_body = f"{prefix}green_{cfg.cube_body}"
        self.red_geom = f"{prefix}red_{cfg.cube_geom}"
        self.green_geom = f"{prefix}green_{cfg.cube_geom}"
        self.red_init_z: float | None = None
        self._gripper_body_ids: list[int] | None = None

    def reset(self):
        super().reset()
        self.red_init_z = None

    @staticmethod
    def _id(sim: Sim, objtype: mj.mjtObj, name: str) -> int:
        i = mj.mj_name2id(sim.model, objtype, name)
        if i == -1:
            msg = f"{objtype!r} '{name}' not found"
            raise ValueError(msg)
        return i

    def _gripper_bodies(self, sim: Sim) -> list[int]:
        if self._gripper_body_ids is None:
            self._gripper_body_ids = [
                i
                for i in range(sim.model.nbody)
                if (
                    (name := sim.model.body(i).name)
                    and (name.startswith("gripper") or "pad" in name or "finger" in name)
                )
            ]
        return self._gripper_body_ids

    def _red_touching_gripper(self, sim: Sim, red_body_id: int) -> bool:
        model, data = sim.model, sim.data
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = int(model.geom_bodyid[c.geom1])
            b2 = int(model.geom_bodyid[c.geom2])
            if red_body_id not in (b1, b2):
                continue
            other = model.body(b2 if b1 == red_body_id else b1).name
            if other.startswith("gripper") or "pad" in other or "finger" in other:
                return True
        return False

    def _min_gripper_distance(self, sim: Sim, red_pos: np.ndarray) -> float:
        ids = self._gripper_bodies(sim)
        if not ids:
            return float("inf")
        return min(float(np.linalg.norm(red_pos - sim.data.xpos[i])) for i in ids)

    def _geoms_in_contact(self, sim: Sim, geom_a: int, geom_b: int) -> bool:
        data = sim.data
        for i in range(data.ncon):
            c = data.contact[i]
            if {int(c.geom1), int(c.geom2)} == {geom_a, geom_b}:
                return True
        return False

    def update_internal_state(self, sim: Sim):
        rb = self._id(sim, mj.mjtObj.mjOBJ_BODY, self.red_body)
        gb = self._id(sim, mj.mjtObj.mjOBJ_BODY, self.green_body)
        rg = self._id(sim, mj.mjtObj.mjOBJ_GEOM, self.red_geom)
        gg = self._id(sim, mj.mjtObj.mjOBJ_GEOM, self.green_geom)
        red = sim.data.xpos[rb]
        green = sim.data.xpos[gb]

        if self.red_init_z is None:
            self.red_init_z = float(red[2])

        s = self.internal_state
        red_in_gripper = self._red_touching_gripper(sim, rb)
        lifted_height = red[2] > self.red_init_z + self.cfg.lift_thresh
        settled = float(np.linalg.norm(sim.data.cvel[rb])) < self.cfg.settle_speed
        on_green = self._geoms_in_contact(sim, rg, gg)
        stacked = on_green and (red[2] - green[2]) > self.cfg.above_thresh and (
            float(np.linalg.norm(red[:2] - green[:2])) < self.cfg.xy_tol
        )
        retracted = (not red_in_gripper) and self._min_gripper_distance(sim, red) > self.cfg.retract_dist

        # monotonic, each stage gated on the previous
        if red_in_gripper:
            s["touched"] = True
        if s["touched"] and red_in_gripper and lifted_height:
            s["lifted"] = True
        if s["lifted"] and on_green:
            s["touched_green"] = True
        if s["touched_green"] and stacked and settled and retracted:
            s["placed"] = True

        self.update_stage()

    def update_stage(self):
        s = self.internal_state
        if s["placed"]:
            self.stage = 4
        elif s["touched_green"]:
            self.stage = 3
        elif s["lifted"]:
            self.stage = 2
        elif s["touched"]:
            self.stage = 1
        else:
            self.stage = 0

    @property
    def info(self) -> dict[str, Any]:
        base = super().info
        base.update(dict(self.internal_state))
        return base


@dataclass(kw_only=True)
class PickPlaceTaskConfig(BaseTaskConfig):
    task_id: str = "pick_place_red_on_green"
    prefix: str = "PickPlace_"
    task_instructions: str = "pick up the red cube and place it on top of the green cube"

    # red_cube.xml / green_cube.xml element names (prefixed with '<prefix>red_' / '<prefix>green_')
    cube_body: str = "box_body"
    cube_geom: str = "box_geom"
    cube_joint: str = "box_joint"

    # success thresholds
    lift_thresh: float = 0.03  # red must rise this much above its start to count as "lifted"
    above_thresh: float = 0.02  # red center must be this far above green center (stacked, not beside)
    xy_tol: float = 0.03  # horizontal alignment of red over green
    settle_speed: float = 0.05  # red spatial-velocity norm below this counts as "at rest"
    retract_dist: float = 0.08  # nearest gripper body must be this far from red to count as "retracted"

    objects_xml: dict[str, str] = field(
        default_factory=lambda: {"red_cube": rcs.OBJECT_PATHS["red_cube"], "green_cube": rcs.OBJECT_PATHS["green_cube"]}
    )
    # initial spawn poses relative to the root (robot-base) frame (overridden each reset by the
    # random placement below); both on the table in front of the arm
    red_center_to_root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(translation=np.array([0.4, -0.1, 0.02]))
    )
    green_center_to_root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(translation=np.array([0.4, 0.1, 0.02]))
    )
    # center of the random-placement square (both cubes placed randomly here each reset), at a
    # height where the cubes rest on the table
    task_center_to_root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(translation=np.array([0.5, 0.0, 0.05]))
    )
    rand_x_width: float = 0.2
    rand_y_width: float = 0.4
    obj_position_margin: float = 0.09  # keep red and green well separated at start


class PickPlaceTask(Task[PickPlaceTaskConfig]):
    @staticmethod
    def add_task_mujoco(cfg: PickPlaceTaskConfig, composer: ModelComposer, env_cfg: SimEnvCreatorConfig):
        """Add the red and green cubes to the scene."""
        red2world = env_cfg.root_frame_to_world * cfg.red_center_to_root_frame
        green2world = env_cfg.root_frame_to_world * cfg.green_center_to_root_frame
        composer.add_object_world_frame(cfg.objects_xml["red_cube"], object_prefix=f"{cfg.prefix}red_", pose=red2world)
        composer.add_object_world_frame(
            cfg.objects_xml["green_cube"], object_prefix=f"{cfg.prefix}green_", pose=green2world
        )

    @staticmethod
    def add_task_env(
        cfg: PickPlaceTaskConfig, env: gym.Env, simulation: Sim, env_cfg: SimEnvCreatorConfig
    ) -> gym.Env:
        _ = simulation
        # randomize both cubes within the task frame on every reset (non-overlapping)
        env = RandomSquareObjsPos(
            env,
            center2world=env_cfg.root_frame_to_world * cfg.task_center_to_root_frame,
            obj_joint_names=[f"{cfg.prefix}red_{cfg.cube_joint}", f"{cfg.prefix}green_{cfg.cube_joint}"],
            x_width=cfg.rand_x_width,
            y_width=cfg.rand_y_width,
            z_init=0.0,
            include_rotation=False,
            obj_position_margin=cfg.obj_position_margin,
        )
        return TaskStageWrapper(env, PickPlaceStage(cfg, prefix=cfg.prefix))


rcs.TASKS.update({"pick_place_red_on_green": PickPlaceTask})


class PickPlaceEnvConfig(SingleArm):
    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.task_cfg = PickPlaceTaskConfig()
        return cfg


gym.register(id="duobench/pick_place_red_on_green", entry_point=PickPlaceEnvConfig())
