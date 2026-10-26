import os
import warnings

import gymnasium as gym
import mujoco as mj
import numpy as np
import pytest
from rcs.envs.scenes import SimEnvCreator

import duobench.tasks.block_balance  # noqa: F401  registers duobench/block_balance
from duobench.tasks.block_balance import BlockBalanceStage, BlockBalanceTaskConfig

warnings.filterwarnings("ignore", category=UserWarning, module=r"gymnasium\.core")

PREFIX = BlockBalanceTaskConfig().prefix
# half heights of the box geoms
CUBE_HALF_Z, BEAM_HALF_Z, RECT_HALF_Z = 0.02, 0.01, 0.045


def test_stages_are_sequential():
    stage = BlockBalanceStage(BlockBalanceTaskConfig(), prefix=PREFIX)
    stage.internal_state["both_cubes_on_beam"] = True
    stage.internal_state["beam_balanced"] = True
    stage.update_stage()
    assert stage.stage == 0

    for key, expected in (
        ("beam_grasped", 1),
        ("beam_on_cube", 2),
        ("both_cubes_grasped", 3),
        ("both_cubes_on_beam", 4),
        ("beam_balanced", 5),
    ):
        stage.internal_state[key] = True
        stage.update_stage()
        assert stage.stage == expected


@pytest.fixture(scope="module")
def env():
    os.environ.setdefault("MUJOCO_GL", "egl")
    env_creator = gym.envs.registry["duobench/block_balance"].entry_point
    assert isinstance(env_creator, SimEnvCreator)
    cfg = env_creator.config()
    cfg.headless = True
    cfg.camera_cfgs = None
    cfg.camera_adds = None
    env = env_creator.create_env(cfg)
    env.reset(seed=0)
    yield env
    env.close()


def _place(sim, name: str, pos: np.ndarray):
    model, data = sim.model, sim.data
    joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, PREFIX + name + "_joint")
    qpos, qvel = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    data.qpos[qpos : qpos + 3] = pos
    data.qpos[qpos + 3 : qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[qvel : qvel + 6] = 0.0


def _settle_and_track(sim, stage: BlockBalanceStage, start_stage: int, steps: int = 300, **state) -> int:
    """Let the objects settle (robots untouched), then run the stage check for a few steps from start_stage
    (with internal_state entries preset from earlier steps)."""
    mj.mj_forward(sim.model, sim.data)
    for _ in range(steps):
        mj.mj_step(sim.model, sim.data)
    stage.reset()
    stage.stage = start_stage
    stage.internal_state.update(state)
    for _ in range(10):
        mj.mj_step(sim.model, sim.data)
        stage.update_internal_state(sim)
    return stage.stage


def _table_and_center(sim) -> tuple[float, np.ndarray]:
    cube = sim.data.body(PREFIX + "big_red_cube").xpos.copy()
    return cube[2] - CUBE_HALF_Z, cube[:2]


def test_balanced_structure_succeeds(env):
    sim = env.get_wrapper_attr("sim")
    env.reset(seed=0)
    table, (x, y) = _table_and_center(sim)
    beam_z = table + 2 * CUBE_HALF_Z + BEAM_HALF_Z
    rect_z = beam_z + BEAM_HALF_Z + RECT_HALF_Z
    _place(sim, "big_red_cube", np.array([x, y, table + CUBE_HALF_Z]))
    _place(sim, "pink_beam", np.array([x, y, beam_z]))
    _place(sim, "blue_rectangle", np.array([x - 0.07, y, rect_z]))
    _place(sim, "green_rectangle", np.array([x + 0.07, y, rect_z]))

    stage = BlockBalanceStage(BlockBalanceTaskConfig(), prefix=PREFIX)
    assert _settle_and_track(sim, stage, start_stage=4) == 5


def test_beam_on_table_is_not_balanced(env):
    """The beam touched the cube once, then lies flat on the table with both rectangles on it: not a success."""
    sim = env.get_wrapper_attr("sim")
    env.reset(seed=0)
    table, (x, y) = _table_and_center(sim)
    beam_z = table + BEAM_HALF_Z
    rect_z = beam_z + BEAM_HALF_Z + RECT_HALF_Z
    _place(sim, "big_red_cube", np.array([x, y + 0.0395, table + CUBE_HALF_Z]))  # touching the beam's side
    _place(sim, "pink_beam", np.array([x, y, beam_z]))
    _place(sim, "blue_rectangle", np.array([x - 0.07, y, rect_z]))
    _place(sim, "green_rectangle", np.array([x + 0.07, y, rect_z]))

    stage = BlockBalanceStage(BlockBalanceTaskConfig(), prefix=PREFIX)
    final = _settle_and_track(sim, stage, start_stage=4)
    assert stage.internal_state["both_cubes_on_beam"]  # the rectangles do stand on the beam
    assert final == 4


def test_beam_on_table_without_cube_contact_is_not_on_cube(env):
    """Old check: beam_on_cube stayed latched after the beam left the cube. New: current contact only."""
    sim = env.get_wrapper_attr("sim")
    env.reset(seed=0)
    table, (x, y) = _table_and_center(sim)
    beam_z = table + BEAM_HALF_Z
    rect_z = beam_z + BEAM_HALF_Z + RECT_HALF_Z
    _place(sim, "big_red_cube", np.array([x, y + 0.15, table + CUBE_HALF_Z]))
    _place(sim, "pink_beam", np.array([x, y, beam_z]))
    _place(sim, "blue_rectangle", np.array([x - 0.07, y, rect_z]))
    _place(sim, "green_rectangle", np.array([x + 0.07, y, rect_z]))

    stage = BlockBalanceStage(BlockBalanceTaskConfig(), prefix=PREFIX)
    final = _settle_and_track(sim, stage, start_stage=4, beam_on_cube=True)  # touched the cube earlier
    assert not stage.internal_state["beam_on_cube"]
    assert not stage.internal_state["both_cubes_on_beam"]
    assert final == 4
