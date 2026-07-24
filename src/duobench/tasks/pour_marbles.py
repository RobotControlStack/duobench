import math
from dataclasses import dataclass, field
from typing import Any, Literal

import gymnasium as gym
import mujoco as mj
import numpy as np
import rcs
from rcs.envs.scenes import BaseTaskConfig, SimEnvCreatorConfig, Task
from rcs.sim.composer import ModelComposer
from rcs.sim.sim import Sim
from scipy.spatial.transform import RigidTransform, Rotation

from duobench.tasks import TaskStage, TaskStageWrapper
from duobench.utils.vention_config import VentionSceneFR3Duo

# Convention for transforms: all transforms are named tf_A_B, which means
# "coordinates of B in frame of A".
# "left" and "right" are with respect to the point of view of the robot, as if you
# are standing on the duo mount base, looking at the workspace.

N_MARBLES = 20
CUP_RADIUS = 0.032
CUP_HEIGHT = 0.105
TARGET_AREA_HALF_SIZE = 0.05


def hexagonal_tiling_in_circle(circle_radius: float, tile_radius: float):
    R = circle_radius - tile_radius
    dx = 2.0 * tile_radius
    dy = math.sqrt(3.0) * tile_radius
    centers = []
    row = 0
    y = -R
    while y <= R:
        x_offset = tile_radius if row % 2 else 0.0
        x = -R
        while x <= R:
            px = x + x_offset
            py = y
            if px * px + py * py <= R * R:
                centers.append((px, py))
            x += dx
        y += dy
        row += 1
    return centers


class PourMarblesStage(TaskStage):
    INSTRUCTION = "grasp and lift both cups, then pour the marbles from one cup into the other and place the cups back to their original location inside the green square"

    def __init__(self):
        super().__init__(
            max_stage=6,
            internal_state={
                "one_cup_grasped": False,
                "both_cups_grasped": False,
                "both_cups_lifted": False,
                "one_marble_in_target_cup": False,
                "all_marbles_in_target_cup": False,
                "cups_placed": False,
                "cups_upright": False,
                "done": False,
            },
            stage_to_subinstructions={
                0: "grasp at least one cup",
                1: "grasp both cups",
                2: "lift both cups",
                3: "pour at least one marble into the target cup",
                4: "pour all marbles into the target cup",
                5: "place both cups",
                6: "task completed; the cups are placed upright",
            },
            instruction=self.INSTRUCTION,
        )
        self.collision_geoms: dict[str, Any] | None = None
        self.source_cup = ""
        self.target_cup = ""
        self.marbles_in_left_cup = 0
        self.marbles_in_right_cup = 0
        self.left_gripper_grasps_left_cup = False
        self.right_gripper_grasps_right_cup = False
        self.left_cup_in_place = True
        self.right_cup_in_place = True
        self.left_cup_lifted = False
        self.right_cup_lifted = False
        self.left_cup_upright = True
        self.right_cup_upright = True

    def set_pour_direction(self, source_cup: Literal["left", "right"], target_cup: Literal["left", "right"]):
        self.source_cup = source_cup
        self.target_cup = target_cup
        self.instruction = (
            f"grasp and lift both cups, then pour the marbles from the {source_cup} cup into the {target_cup} cup "
            "and place the cups back to their original location inside the green square"
        )
        self.stage_to_subinstructions[3] = f"pour at least one marble from the {source_cup} cup into the {target_cup} cup"
        self.stage_to_subinstructions[4] = f"pour all marbles from the {source_cup} cup into the {target_cup} cup"

    def _ensure_collision_geoms(self, sim: Sim):
        if self.collision_geoms is not None:
            return
        self.collision_geoms = {
            "gripper": {
                "left": {
                    "left": {
                        mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_GEOM, "gripperleft_left_pad1"),
                        mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_GEOM, "gripperleft_left_pad2"),
                    },
                    "right": {
                        mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_GEOM, "gripperleft_right_pad1"),
                        mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_GEOM, "gripperleft_right_pad2"),
                    },
                },
                "right": {
                    "left": {
                        mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_GEOM, "gripperright_left_pad1"),
                        mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_GEOM, "gripperright_left_pad2"),
                    },
                    "right": {
                        mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_GEOM, "gripperright_right_pad1"),
                        mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_GEOM, "gripperright_right_pad2"),
                    },
                },
            },
            "cup": {
                "left": set(
                    np.flatnonzero(
                        sim.model.geom_bodyid == mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, "leftteacup_body")
                    )
                ),
                "right": set(
                    np.flatnonzero(
                        sim.model.geom_bodyid == mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, "rightteacup_body")
                    )
                ),
            },
            "marbles": set(
                np.flatnonzero(
                    np.isin(
                        sim.model.geom_bodyid,
                        [mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, f"{i}marble") for i in range(N_MARBLES)],
                    )
                )
            ),
        }

    def touching(self, a: set[int], b: set[int]) -> bool:
        raise NotImplementedError

    def _touching(self, sim: Sim, a: set[int], b: set[int]) -> bool:
        return any((c.geom1 in a and c.geom2 in b) or (c.geom2 in a and c.geom1 in b) for c in sim.data.contact)

    def reset(self):
        super().reset()
        self.source_cup = ""
        self.target_cup = ""
        self.marbles_in_left_cup = 0
        self.marbles_in_right_cup = 0
        self.left_gripper_grasps_left_cup = False
        self.right_gripper_grasps_right_cup = False
        self.left_cup_in_place = True
        self.right_cup_in_place = True
        self.left_cup_lifted = False
        self.right_cup_lifted = False
        self.left_cup_upright = True
        self.right_cup_upright = True

    def _infer_cup_roles(self, marbles_in_cups: dict[str, set[int]]):
        if len(marbles_in_cups["left"]) == len(marbles_in_cups["right"]):
            return

        progress_started = any(
            self.internal_state[key]
            for key in (
                "one_cup_grasped",
                "both_cups_grasped",
                "both_cups_lifted",
                "one_marble_in_target_cup",
                "all_marbles_in_target_cup",
                "cups_placed",
                "done",
            )
        )

        if progress_started and self.source_cup and self.target_cup:
            return

        self.source_cup = "left" if len(marbles_in_cups["left"]) > len(marbles_in_cups["right"]) else "right"
        self.target_cup = "right" if self.source_cup == "left" else "left"

    def update_internal_state(self, sim: Sim):
        self._ensure_collision_geoms(sim)
        assert self.collision_geoms is not None
        lc = sim.data.body("leftteacup_body")
        rc = sim.data.body("rightteacup_body")
        lt = sim.data.geom("left_target")
        rt = sim.data.geom("right_target")
        ms = [sim.data.body(f"{i}marble") for i in range(N_MARBLES)]
        tf_W_LC = RigidTransform.from_components(lc.xpos, Rotation.from_matrix(lc.xmat.reshape((3, 3))))
        tf_W_RC = RigidTransform.from_components(rc.xpos, Rotation.from_matrix(rc.xmat.reshape((3, 3))))
        tf_W_Ms = [RigidTransform.from_components(m.xpos, Rotation.from_matrix(m.xmat.reshape((3, 3)))) for m in ms]
        tf_W_LT = RigidTransform.from_components(lt.xpos, Rotation.from_matrix(lt.xmat.reshape((3, 3))))
        tf_W_RT = RigidTransform.from_components(rt.xpos, Rotation.from_matrix(rt.xmat.reshape((3, 3))))
        tf_LT_LC = tf_W_LT.inv() * tf_W_LC
        tf_RT_RC = tf_W_RT.inv() * tf_W_RC
        marbles_in_cups = {
            name: {
                i
                for i, tf in enumerate(tf_W_C.inv() * tf_W_M for tf_W_M in tf_W_Ms)
                if np.linalg.norm(tf.translation[:2]) < CUP_RADIUS
                and tf.translation[-1] < CUP_HEIGHT
                and tf.translation[-1] > 0
            }
            for (name, tf_W_C) in zip(("left", "right"), (tf_W_LC, tf_W_RC), strict=True)
        }
        self._infer_cup_roles(marbles_in_cups)
        left_gripper_grips_left_cup = self._touching(
            sim, self.collision_geoms["gripper"]["left"]["left"], self.collision_geoms["cup"]["left"]
        ) and self._touching(sim, self.collision_geoms["gripper"]["left"]["right"], self.collision_geoms["cup"]["left"])
        right_gripper_grips_right_cup = self._touching(
            sim, self.collision_geoms["gripper"]["right"]["left"], self.collision_geoms["cup"]["right"]
        ) and self._touching(
            sim, self.collision_geoms["gripper"]["right"]["right"], self.collision_geoms["cup"]["right"]
        )
        self.marbles_in_left_cup = len(marbles_in_cups["left"])
        self.marbles_in_right_cup = len(marbles_in_cups["right"])
        self.left_gripper_grasps_left_cup = left_gripper_grips_left_cup
        self.right_gripper_grasps_right_cup = right_gripper_grips_right_cup
        self.left_cup_in_place = (
            abs(tf_LT_LC.translation[0]) < TARGET_AREA_HALF_SIZE
            and abs(tf_LT_LC.translation[1]) < TARGET_AREA_HALF_SIZE
            and tf_LT_LC.translation[2] < 0.001
            and tf_LT_LC.translation[2] >= 0
        )
        self.right_cup_in_place = (
            abs(tf_RT_RC.translation[0]) < TARGET_AREA_HALF_SIZE
            and abs(tf_RT_RC.translation[1]) < TARGET_AREA_HALF_SIZE
            and tf_RT_RC.translation[2] < 0.001
            and tf_RT_RC.translation[2] >= 0
        )
        self.left_cup_lifted = tf_LT_LC.translation[2] > 0.001
        self.right_cup_lifted = tf_RT_RC.translation[2] > 0.001
        world_up = np.array([0.0, 0.0, 1.0])
        self.left_cup_upright = float(np.dot(tf_W_LC.rotation.as_matrix()[:, 2], world_up)) > math.cos(math.radians(20))
        self.right_cup_upright = float(np.dot(tf_W_RC.rotation.as_matrix()[:, 2], world_up)) > math.cos(
            math.radians(20)
        )

        one_cup_grasped = self.left_gripper_grasps_left_cup or self.right_gripper_grasps_right_cup
        both_cups_grasped = self.left_gripper_grasps_left_cup and self.right_gripper_grasps_right_cup
        both_cups_lifted = self.left_cup_lifted and self.right_cup_lifted
        marbles_in_target_cup = self.marbles_in_left_cup if self.target_cup == "left" else self.marbles_in_right_cup
        one_marble_in_target_cup = marbles_in_target_cup >= 1
        all_marbles_in_target_cup = marbles_in_target_cup == N_MARBLES
        cups_placed = self.left_cup_in_place and self.right_cup_in_place
        cups_upright = self.left_cup_upright and self.right_cup_upright
        done_now = cups_placed and cups_upright

        self.internal_state["one_cup_grasped"] = self.internal_state["one_cup_grasped"] or one_cup_grasped
        self.internal_state["both_cups_grasped"] = self.internal_state["both_cups_grasped"] or both_cups_grasped
        self.internal_state["both_cups_lifted"] = self.internal_state["both_cups_lifted"] or both_cups_lifted
        self.internal_state["one_marble_in_target_cup"] = (
            self.internal_state["one_marble_in_target_cup"] or one_marble_in_target_cup
        )
        self.internal_state["all_marbles_in_target_cup"] = (
            self.internal_state["all_marbles_in_target_cup"] or all_marbles_in_target_cup
        )
        self.internal_state["cups_placed"] = self.internal_state["cups_placed"] or (
            self.internal_state["all_marbles_in_target_cup"] and cups_placed
        )
        self.internal_state["cups_upright"] = cups_upright
        self.internal_state["done"] = self.internal_state["done"] or (self.internal_state["cups_placed"] and done_now)

        self.update_stage()

    def update_stage(self):
        if self.stage == 0 and self.internal_state["one_cup_grasped"]:
            self.stage = 1
        elif self.stage == 1 and self.internal_state["both_cups_grasped"]:
            self.stage = 2
        elif self.stage == 2 and self.internal_state["both_cups_lifted"]:
            self.stage = 3
        elif self.stage == 3 and self.internal_state["one_marble_in_target_cup"]:
            self.stage = 4
        elif self.stage == 4 and self.internal_state["all_marbles_in_target_cup"]:
            self.stage = 5
        elif self.stage == 5 and self.internal_state["done"]:
            self.stage = 6

    @property
    def info(self) -> dict[str, Any]:
        return super().info | {
            "source_cup": self.source_cup,
            "target_cup": self.target_cup,
            "marbles_in_left_cup": self.marbles_in_left_cup,
            "marbles_in_right_cup": self.marbles_in_right_cup,
            "left_gripper_grasps_left_cup": self.left_gripper_grasps_left_cup,
            "right_gripper_grasps_right_cup": self.right_gripper_grasps_right_cup,
            "left_cup_in_place": self.left_cup_in_place,
            "right_cup_in_place": self.right_cup_in_place,
            "left_cup_lifted": self.left_cup_lifted,
            "right_cup_lifted": self.right_cup_lifted,
            "left_cup_upright": self.left_cup_upright,
            "right_cup_upright": self.right_cup_upright,
        }


class PourMarblesTaskWrapper(TaskStageWrapper):
    def __init__(self, env: gym.Env, stage_tracker: PourMarblesStage):
        super().__init__(env, stage_tracker)
        self.stage_tracker: PourMarblesStage = stage_tracker
        self.sim = self.get_wrapper_attr("sim")
        task_cfg = self.get_wrapper_attr("task_cfg")
        self.marble_spawn_cup: Literal["random", "left", "right"] = task_cfg.marble_spawn_cup

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)

        if self.marble_spawn_cup == "random":
            source_cup: Literal["left", "right"] = "left" if self.np_random.random() > 0.5 else "right"
        else:
            source_cup = self.marble_spawn_cup

        target_cup: Literal["left", "right"] = "right" if source_cup == "left" else "left"
        self.stage_tracker.set_pour_direction(source_cup, target_cup)
        lc = self.sim.data.body("leftteacup_body")
        rc = self.sim.data.body("rightteacup_body")
        tf_W_LC = RigidTransform.from_components(lc.xpos, Rotation.from_matrix(lc.xmat.reshape((3, 3))))
        tf_W_RC = RigidTransform.from_components(rc.xpos, Rotation.from_matrix(rc.xmat.reshape((3, 3))))
        off_x_l, off_y_l, off_x_r, off_y_r = (
            (self.np_random.random() - 0.5) * 0.1,
            (self.np_random.random() - 0.5) * 0.1,
            (self.np_random.random() - 0.5) * 0.1,
            (self.np_random.random() - 0.5) * 0.1,
        )
        self.sim.data.joint("leftteacup_joint").qpos[0] += off_x_l
        self.sim.data.joint("leftteacup_joint").qpos[1] += off_y_l
        self.sim.data.joint("rightteacup_joint").qpos[0] += off_x_r
        self.sim.data.joint("rightteacup_joint").qpos[1] += off_y_r
        for i in range(N_MARBLES):
            tf_W_M = RigidTransform.from_components(
                self.sim.data.body(f"{i}marble").xpos,
                Rotation.from_matrix(self.sim.data.body(f"{i}marble").xmat.reshape((3, 3))),
            )
            tf_W_C = tf_W_LC if self.stage_tracker.source_cup == "left" else tf_W_RC
            off_x, off_y = (off_x_l, off_y_l) if self.stage_tracker.source_cup == "left" else (off_x_r, off_y_r)
            spawn_position = tf_W_M * tf_W_C
            self.sim.data.joint(f"{i}marble").qpos[:3] = spawn_position.translation
            self.sim.data.joint(f"{i}marble").qpos[0] += off_x
            self.sim.data.joint(f"{i}marble").qpos[1] += off_y
        self.stage_tracker.marbles_in_left_cup = N_MARBLES if self.stage_tracker.source_cup == "left" else 0
        self.stage_tracker.marbles_in_right_cup = N_MARBLES if self.stage_tracker.source_cup == "right" else 0
        mj.mj_forward(self.sim.model, self.sim.data)
        self.stage_tracker.update_internal_state(self.sim)
        info.update(self.stage_tracker.info)
        return obs, info


@dataclass(kw_only=True)
class PourMarblesTaskConfig(BaseTaskConfig):
    task_id: str = "pour_marbles"
    marble_spawn_cup: Literal["random", "left", "right"] = "right"
    cup_xml = rcs.OBJECT_PATHS["teacup"]
    marble_xml = rcs.OBJECT_PATHS["marble"]
    marbles_to_mug: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(
            translation=np.array([0.0, 0.0, 0.02]), quaternion=np.array([0, 0, 0, 1])
        )
    )
    mug_to_root: dict[str, rcs.common.Pose] = field(
        default_factory=lambda: {
            "right": rcs.common.Pose(
                translation=np.array([0.5, -0.3, 0.0125]),
                quaternion=np.array([0, 0, -math.sqrt(2) / 2, math.sqrt(2) / 2]),
            ),
            "left": rcs.common.Pose(
                translation=np.array([0.5, 0.3, 0.0125]),
                quaternion=np.array([0, 0, math.sqrt(2) / 2, math.sqrt(2) / 2]),
            ),
        }
    )


class PourMarblesTask(Task[PourMarblesTaskConfig]):
    @staticmethod
    def add_task_mujoco(cfg: PourMarblesTaskConfig, composer: ModelComposer, env_cfg: SimEnvCreatorConfig):
        """Add task-specific elements to the Mujoco scene."""
        for mug_id, mug_to_root in cfg.mug_to_root.items():
            composer.add_object_world_frame(
                cfg.cup_xml,
                object_prefix=mug_id,
                pose=mug_to_root * env_cfg.root_frame_to_world,
                register_root_relative_replay_free_joints=True,
            )
            geom = composer.spec.worldbody.add_geom()
            geom.name = f"{mug_id}_target"
            geom.type = mj.mjtGeom.mjGEOM_BOX
            geom.pos[:] = (mug_to_root * env_cfg.root_frame_to_world).translation()
            geom.size[:] = 0.1, 0.1, 0.03
            geom.rgba[:] = 0, 1, 0, 0.1
            geom.group = 2
            geom.contype = 0
            geom.conaffinity = 0
        marble_centers = hexagonal_tiling_in_circle(CUP_RADIUS - 0.001, 0.005)
        for idx in range(N_MARBLES):
            marble2world = rcs.common.Pose(
                translation=np.concatenate([marble_centers[idx], np.array([0.02])]), quaternion=np.array([0, 0, 0, 1])
            )
            composer.add_object_world_frame(
                cfg.marble_xml,
                object_prefix=str(idx),
                pose=marble2world,
                register_root_relative_replay_free_joints=True,
            )

    @staticmethod
    def add_task_env(
        cfg: PourMarblesTaskConfig, env: gym.Env, simulation: Sim, env_cfg: SimEnvCreatorConfig
    ) -> gym.Env:
        """Add task-specific wrappers to the environment."""
        _ = cfg, simulation, env_cfg
        return PourMarblesTaskWrapper(env, PourMarblesStage())


rcs.TASKS["pour_marbles"] = PourMarblesTask


class PourMarblesEnvConfig(VentionSceneFR3Duo):

    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.task_cfg = PourMarblesTaskConfig()
        return cfg


gym.register(id="duobench/pour_marbles", entry_point=PourMarblesEnvConfig())
