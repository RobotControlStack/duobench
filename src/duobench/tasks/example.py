from dataclasses import dataclass

import gymnasium as gym
import rcs
from rcs.envs.scenes import BaseTaskConfig, SimEnvCreatorConfig, Task
from rcs.sim.composer import ModelComposer
from rcs.sim.sim import Sim

from duobench.utils.vention_config import VentionSceneFR3Duo


@dataclass(kw_only=True)
class MyTaskConfig(BaseTaskConfig):
    task_id: str = "mytask"


class MyTask(Task[MyTaskConfig]):

    @staticmethod
    def add_task_mujoco(cfg: MyTaskConfig, composer: ModelComposer, env_cfg: SimEnvCreatorConfig):
        """Add task-specific elements to the Mujoco scene."""
        _ = cfg, composer, env_cfg

    @staticmethod
    def add_task_env(cfg: MyTaskConfig, env: gym.Env, simulation: Sim, env_cfg: SimEnvCreatorConfig) -> gym.Env:
        """Add task-specific wrappers to the environment."""
        _ = cfg, simulation, env_cfg
        return env


# we add our task classes here
rcs.TASKS.update({"mytask": MyTask})


class MyTaskEnvConfig(VentionSceneFR3Duo):

    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.task_cfg = MyTaskConfig()
        return cfg


gym.register(id="duobench/example", entry_point=MyTaskEnvConfig())
