import numpy as np

from duobench.tasks.ball_maze import BallMazeStage, BallMazeTaskConfig


def test_ball_maze_start_distance_ignores_vertical_offset():
    stage = BallMazeStage(BallMazeTaskConfig())

    start_pose = np.array([0.0, 0.0, 0.0033])
    ball_pose = np.array([0.0, 0.0, 0.05])

    assert stage._planar_distance(start_pose, ball_pose) == 0.0
