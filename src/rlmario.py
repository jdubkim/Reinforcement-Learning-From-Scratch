import gym

import setting
import config

setting.set_env()
env = gym.make('SuperMarioBros-1-1-v0')
env.reset()

for _ in range(config.nb_iter):
    env.render()
    env.step(env.action_space.sample()) # take a random action

