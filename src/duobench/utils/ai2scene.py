import time

from duobench.tasks.flipping import FlippingEnvConfig


if __name__ == "__main__":
    scene = FlippingEnvConfig()
    env = scene.create_env(scene.config())
    obs, info = env.reset()
    print(obs)
    # Duo
    for _ in range(100):
        for _ in range(10):
            # move 1cm in x direction (forward) and close gripper
            act = {
                "right": {"tquat": [0.0, 0, 0, 0, 0, 0, 1], "gripper": [0]},
            }
            obs, reward, terminated, truncated, info = env.step(act)
            # print(obs)
            print(reward, terminated, truncated, info)
            time.sleep(0.033)
        for _ in range(10):
            # move 1cm in negative x direction (backward) and open gripper
            act = {
                "right": {"tquat": [-0.0, 0, 0, 0, 0, 0, 1], "gripper": [1]},
            }
            obs, reward, terminated, truncated, info = env.step(act)
            # print(obs)
            print(reward, terminated, truncated, info)
            time.sleep(0.033)