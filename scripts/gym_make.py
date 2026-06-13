import rcs
from duobench.tasks import ball_maze
from duobench.tasks import bin_sort
from duobench.tasks import block_balance
from duobench.tasks import carry_pot
from duobench.tasks import join_blocks
from duobench.tasks import hinge_chest
from duobench.tasks import pour_marbles
from duobench.tasks import spring_door
from duobench.tasks import transfer_cube
from duobench.tasks import transfer_gate
from duobench.tasks import transfer_reorient
import gymnasium as gym
import time

def main():
    cfg = ball_maze.BallMazeEnvConfig().config()
    cfg.sim_cfg.realtime = True
    cfg.sim_cfg.async_control = False
    cfg.sim_cfg.frequency = 1
    cfg.headless = False
    env = gym.make("duobench/ball_maze", cfg=cfg)
    # env = ball_maze.BallMazeEnvConfig().create_env(cfg)
    time.sleep(5)

    try:
        env.reset()
        for _ in range(100):
            for _ in range(10):
                # move 1cm in x direction (forward) and close gripper
                act = {
                    "left": {"tquat": [0.01, 0, 0, 0, 0, 0, 1], "gripper": [0]},
                    "right": {"tquat": [0.01, 0, 0, 0, 0, 0, 1], "gripper": [0]},
                }
                obs, reward, terminated, truncated, info = env.step(act)
                print(obs)
                time.sleep(1)
            for _ in range(10):
                # move 1cm in negative x direction (backward) and open gripper
                act = {
                    "left": {"tquat": [-0.01, 0, 0, 0, 0, 0, 1], "gripper": [1]},
                    "right": {"tquat": [-0.01, 0, 0, 0, 0, 0, 1], "gripper": [1]},
                }
                obs, reward, terminated, truncated, info = env.step(act)
                print(obs)
                time.sleep(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
