import importlib
import os
import pkgutil
import warnings
from pathlib import Path

import duckdb
import gymnasium as gym
import pytest
from rcs._core.sim import SimConfig
from rcs.envs.base import RelativeTo
from rcs.envs.scenes import SimEnvCreator
from rcs.sim.replayer import load_trajectory, restore_sim_step

import duobench.tasks

warnings.filterwarnings("ignore", category=UserWarning, module=r"gymnasium\.core")

for module in pkgutil.iter_modules(duobench.tasks.__path__):
    if module.name != "example":
        importlib.import_module(f"duobench.tasks.{module.name}")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPLAY_DATA_ROOT = REPO_ROOT / "test_data" / "replay"
REGISTERED_TASK_IDS = sorted(env_id.split("/", 1)[1] for env_id in gym.envs.registry if env_id.startswith("duobench/"))
TASK_IDS = (
    sorted(path.name for path in REPLAY_DATA_ROOT.iterdir() if path.is_dir() and path.name in REGISTERED_TASK_IDS)
    if REPLAY_DATA_ROOT.exists()
    else []
)

if not TASK_IDS:
    pytestmark = pytest.mark.skip(reason=f"No replay datasets found at {REPLAY_DATA_ROOT}")


REQUIRED_STAGES_BY_TASK = {
    # bin_sort allows successful trajectories that skip stage 2 (both cubes picked)
    # and/or stage 4 (second cube picked after the first correct placement),
    # depending on whether the episode places one cube before picking the other.
    "bin_sort": {0, 1, 3, 5},
    # ball_maze can advance from both-arm contact directly to "in transit" when the
    # replayed step first observed after a restore already has the ball outside the start area.
    "ball_maze": {0, 1, 3, 4},
}


def _load_successful_uuid(dataset_dir: Path) -> str:
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            "SELECT DISTINCT uuid FROM read_parquet(?) WHERE success ORDER BY uuid",
            [str(dataset_dir / "*.parquet")],
        ).fetchall()
    finally:
        connection.close()

    assert rows, f"No successful episodes found in {dataset_dir}"
    return str(rows[0][0])


def _make_replay_env(task_id: str) -> gym.Env:
    os.environ.setdefault("MUJOCO_GL", "egl")
    env_id = f"duobench/{task_id}"
    env_creator = gym.envs.registry[env_id].entry_point
    assert isinstance(env_creator, SimEnvCreator), f"Unexpected env creator for {env_id}: {type(env_creator)!r}"

    cfg = env_creator.config()
    cfg.sim_cfg = SimConfig(async_control=True, realtime=False, frequency=30, max_convergence_steps=500)
    cfg.headless = True
    cfg.relative_to = RelativeTo.CONFIGURED_ORIGIN
    cfg.camera_cfgs = None
    cfg.camera_adds = None
    if cfg.root_frame_objects is None:
        cfg.root_frame_objects = {}

    return env_creator.create_env(cfg)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_successful_replay_visits_all_stages_and_succeeds(task_id: str):
    dataset_dir = REPLAY_DATA_ROOT / task_id
    successful_uuid = _load_successful_uuid(dataset_dir)
    recorded_steps = load_trajectory(dataset_dir, successful_uuid)

    assert recorded_steps, f"No recorded steps found for successful episode {successful_uuid}"

    env = _make_replay_env(task_id)
    try:
        _, info = env.reset()
        assert "stage" in info
        assert "max_stage" in info
        assert "current_subinstruction" in info
        assert "stage_to_subinstructions" in info

        stages_seen = {info["stage"]}
        max_stage = int(info["max_stage"])
        terminated = bool(info["success"])

        for recorded_step in recorded_steps:
            restore_sim_step(env, recorded_step)
            _, _, terminated, truncated, info = env.step(recorded_step.action)

            assert not truncated, f"Replay unexpectedly truncated for {task_id}"
            assert "stage" in info
            assert "max_stage" in info
            assert "current_subinstruction" in info
            assert "stage_to_subinstructions" in info

            stages_seen.add(int(info["stage"]))
            max_stage = int(info["max_stage"])

        all_valid_stages = set(range(max_stage + 1))
        required_stages = REQUIRED_STAGES_BY_TASK.get(task_id, all_valid_stages)
        assert stages_seen <= all_valid_stages, (
            f"{task_id} reported invalid stages during replay. "
            f"Expected only stages within {sorted(all_valid_stages)}, saw {sorted(stages_seen)}"
        )
        assert required_stages <= stages_seen, (
            f"{task_id} did not visit all required stages during replay. "
            f"Required {sorted(required_stages)}, saw {sorted(stages_seen)}"
        )
        assert info["success"], f"{task_id} did not report success after replaying a successful episode"
        assert terminated, f"{task_id} did not terminate after replaying a successful episode"
    finally:
        env.close()
