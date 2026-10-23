import time

from duobench.utils.flexiv_mount_config import FlexivMountRizon4SDuo


if __name__ == "__main__":
    # two Rizon 4s on the Flexiv dual arm mount
    scene = FlexivMountRizon4SDuo()
    env = scene.create_env(scene.config())
    obs, info = env.reset()
    print(obs)
    try:
        while True:
            for _ in range(10):
                # move 1cm in x direction (forward) and close gripper
                act = {
                    "left": {"tquat": [0.01, 0, 0, 0, 0, 0, 1], "gripper": [0]},
                    "right": {"tquat": [0.01, 0, 0, 0, 0, 0, 1], "gripper": [0]},
                }
                obs, reward, terminated, truncated, info = env.step(act)
                time.sleep(0.1)
            for _ in range(10):
                # move 1cm in negative x direction (backward) and open gripper
                act = {
                    "left": {"tquat": [-0.01, 0, 0, 0, 0, 0, 1], "gripper": [1]},
                    "right": {"tquat": [-0.01, 0, 0, 0, 0, 0, 1], "gripper": [1]},
                }
                obs, reward, terminated, truncated, info = env.step(act)
                time.sleep(0.1)
    finally:
        env.close()
