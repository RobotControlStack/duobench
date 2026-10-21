import pytest
from rcs._core.sim import SimConfig

from duobench.tasks.ball_maze import BallMazeEnvConfig
from duobench.utils.vention_config import MUJOCO_MEMORY


def _arena_size(sim_cfg: SimConfig | None = None) -> int:
    scene = BallMazeEnvConfig()
    cfg = scene.config()
    cfg.headless = True
    if sim_cfg is not None:
        cfg.sim_cfg = sim_cfg
    env = scene.create_env(cfg)
    try:
        return env.get_wrapper_attr("sim").model.narena
    finally:
        env.close()


def test_default_config_uses_large_arena():
    assert _arena_size() == MUJOCO_MEMORY


def test_replaced_sim_config_keeps_large_arena():
    # e.g. eval harnesses that set their own SimConfig
    assert _arena_size(SimConfig(async_control=True, frequency=30)) == MUJOCO_MEMORY


@pytest.mark.parametrize("memory", [64 * 1024 * 1024])
def test_explicit_memory_is_kept(memory: int):
    assert _arena_size(SimConfig(memory=memory)) == memory
