import gymnasium as gym
import numpy as np
import rcs

from duobench.utils.helper_wrappers import RandomSquareObjsPos


class _FakeJoint:
    def __init__(self, qpos: np.ndarray):
        self.qpos = qpos


class _FakeData:
    def __init__(self, joints: dict[str, _FakeJoint]):
        self._joints = joints

    def joint(self, name: str) -> _FakeJoint:
        return self._joints[name]


class _FakeSim:
    def __init__(self, joints: dict[str, _FakeJoint]):
        self.data = _FakeData(joints)


class _DummyEnv(gym.Env):
    metadata = {}

    def __init__(self, sim: _FakeSim):
        super().__init__()
        self.sim = sim

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {}, {}

    def step(self, action):
        raise NotImplementedError


class _FakeRng:
    def __init__(self, uniform_values: list[float]):
        self._uniform_values = iter(uniform_values)

    def uniform(self, low, high):
        _ = low, high
        return next(self._uniform_values)

    def random(self):
        return 1.0


class _DeterministicRandomSquareObjsPos(RandomSquareObjsPos):
    def __init__(self, *args, fake_rng: _FakeRng, **kwargs):
        super().__init__(*args, **kwargs)
        self._fake_rng = fake_rng

    @property
    def np_random(self):
        return self._fake_rng


def test_random_square_objs_pos_resamples_away_from_avoided_joint():
    avoided_joint_name = "goal_joint"
    object_joint_name = "cube_joint"
    joints = {
        avoided_joint_name: _FakeJoint(np.array([0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0])),
        object_joint_name: _FakeJoint(np.zeros(7)),
    }
    env = _DummyEnv(_FakeSim(joints))
    wrapper = _DeterministicRandomSquareObjsPos(
        env,
        fake_rng=_FakeRng([0.0, 0.0, 0.09, 0.09]),
        center2world=rcs.common.Pose(
            translation=np.zeros(3),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        ),
        obj_joint_names=[object_joint_name],
        avoid_joint_names=[avoided_joint_name],
        avoid_position_margin=0.05,
        x_width=0.2,
        y_width=0.2,
        z_init=0.02,
        include_rotation=False,
    )

    wrapper.reset()

    spawned_pos = joints[object_joint_name].qpos[:3]
    assert np.allclose(spawned_pos, np.array([0.09, 0.09, 0.02]))
    assert np.linalg.norm(spawned_pos[:2] - joints[avoided_joint_name].qpos[:2]) >= 0.05
