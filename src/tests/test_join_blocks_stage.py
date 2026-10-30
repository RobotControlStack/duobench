import os
import warnings

import gymnasium as gym
import mujoco as mj
import numpy as np
import pytest
from rcs.envs.scenes import SimEnvCreator

import duobench.tasks.join_blocks  # noqa: F401  registers duobench/join_blocks
from duobench.tasks.join_blocks import (
    JoinBlocksStage,
    JoinBlocksTaskConfig,
    get_geom_xmat,
    peg_in_socket,
)
from duobench.utils.helper_wrappers import get_geom_pos

warnings.filterwarnings("ignore", category=UserWarning, module=r"gymnasium\.core")

# p_block orientation relative to h_block with the peg in the socket (from the successful sim demos)
P_IN_H_QUAT = np.array([np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)])


def test_peg_in_socket_uses_marker_frame():
    marker_xmat = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 deg about z
    marker = np.array([0.5, 0.2, 0.1])

    assert peg_in_socket(marker + np.array([0.001, -0.002, 0.012]), marker, marker_xmat, 0.015)
    assert not peg_in_socket(marker + np.array([0.03, 0.0, 0.0]), marker, marker_xmat, 0.015)
    assert not peg_in_socket(marker + np.array([0.0, 0.0, 0.02]), marker, marker_xmat, 0.015)


@pytest.fixture(scope="module")
def env():
    os.environ.setdefault("MUJOCO_GL", "egl")
    env_creator = gym.envs.registry["duobench/join_blocks"].entry_point
    assert isinstance(env_creator, SimEnvCreator)
    cfg = env_creator.config()
    cfg.headless = True
    cfg.camera_cfgs = None
    cfg.camera_adds = None
    env = env_creator.create_env(cfg)
    env.reset(seed=0)
    yield env
    env.close()


def _place_p_block(sim, cfg: JoinBlocksTaskConfig, offset_in_marker_frame: np.ndarray) -> float:
    """Put the p_block peg at the h_block socket marker (+ offset); returns the deepest block penetration."""
    model, data = sim.model, sim.data
    h_block = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg.h_block)
    p_block = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, cfg.p_block)
    qpos = model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, cfg.p_block_joint)]

    mj.mju_mulQuat(data.qpos[qpos + 3 : qpos + 7], data.xquat[h_block], P_IN_H_QUAT)
    for _ in range(2):  # orientation first, then shift the peg center onto the target
        mj.mj_forward(model, data)
        marker_xmat = get_geom_xmat(model, data, cfg.marker_block).reshape(3, 3)
        target = get_geom_pos(model, data, cfg.marker_block) + marker_xmat @ offset_in_marker_frame
        data.qpos[qpos : qpos + 3] += target - get_geom_pos(model, data, cfg.block_peg_name)
    mj.mj_forward(model, data)

    h_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == h_block and model.geom_contype[g]]
    p_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == p_block and model.geom_contype[g]]
    fromto = np.zeros(6)
    return min(mj.mj_geomDistance(model, data, h, p, 0.05, fromto) for h in h_geoms for p in p_geoms)


@pytest.mark.parametrize(
    ("offset", "connected"),
    [
        ((0.0, 0.0, 0.0), True),  # peg fully inserted
        ((0.0, 0.0, 0.01), True),  # peg 1 cm short of full insertion depth
        ((0.035, 0.0, 0.0), False),  # blocks side by side, peg next to the socket
        ((0.0, -0.03, 0.0), False),  # blocks side by side, peg in front of the socket opening
        ((0.0, 0.0, 0.03), False),  # peg on top of the socket
    ],
)
def test_blocks_connected_requires_inserted_peg(env, offset, connected):
    cfg = JoinBlocksTaskConfig()
    sim = env.unwrapped.sim
    penetration = _place_p_block(sim, cfg, np.array(offset))
    assert penetration > -0.003, "test pose is not physically reachable"

    stage = JoinBlocksStage(cfg)
    stage.update_internal_state(sim)
    assert stage.internal_state["blocks_connected"] is connected

    # the previous check (5 cm on every world axis) counted all of these as connected
    diff = np.abs(
        get_geom_pos(sim.model, sim.data, cfg.block_peg_name) - get_geom_pos(sim.model, sim.data, cfg.marker_block)
    )
    assert np.all(diff < 0.05)
