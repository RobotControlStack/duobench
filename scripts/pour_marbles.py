from duobench.tasks.pour_marbles import PourMarblesEnvConfig

if __name__ == "__main__" :
    scene = PourMarblesEnvConfig()
    env = scene.create_env(scene.config())
    obs, info = env.reset()
    for _ in range(100):
        env.step(env.action_space.sample())
