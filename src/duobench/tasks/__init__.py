from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
from rcs import sim as simulation


@dataclass(kw_only=True)
class TaskStage(ABC):
    stage: int = 0
    max_stage: int = 0
    internal_state: dict[str, bool] = field(default_factory=dict)
    stage_to_subinstructions: dict[int, str] = field(default_factory=dict)
    instruction: str = ""

    @abstractmethod
    def update_stage(self):
        msg = "TaskStage is an abstract class. Please implement the update_stage method in a subclass."
        raise NotImplementedError(msg)

    @abstractmethod
    def update_internal_state(self, sim: simulation.Sim):
        msg = "TaskStage is an abstract class. Please implement the update_internal_state method in a subclass."
        raise NotImplementedError(msg)

    def reset(self):
        self.stage = 0
        for key in self.internal_state:
            self.internal_state[key] = False

    @property
    def current_subinstruction(self) -> str:
        return self.stage_to_subinstructions[self.stage] if self.stage_to_subinstructions is not None else None

    @property
    def success(self) -> bool:
        return self.stage == self.max_stage

    @property
    def normalized_stage(self) -> float:
        return self.stage / self.max_stage

    @property
    def info(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "stage": self.stage,
            "max_stage": self.max_stage,
            "current_subinstruction": self.current_subinstruction,
            "stage_to_subinstructions": {
                str(stage): instruction for stage, instruction in self.stage_to_subinstructions.items()
            },
            "instruction": self.instruction,
        }


class TaskStageWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        stage_tracker: TaskStage,
    ):
        super().__init__(env)
        self.stage_tracker = stage_tracker

    def step(self, action: dict[str, Any]):
        obs, _, _, truncated, info = super().step(action)
        self.stage_tracker.update_internal_state(self.get_wrapper_attr("sim"))

        info.update(self.stage_tracker.info)
        reward = self.stage_tracker.normalized_stage

        return obs, reward, self.stage_tracker.success, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.stage_tracker.reset()
        obs, info = super().reset(seed=seed, options=options)
        info.update(self.stage_tracker.info)
        return obs, info
