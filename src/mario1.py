import gym
from src.setting import *

env_set()
env = gym.make('SuperMarioBros-1-1-v0')
env.reset()


for _ in range(10000):
    env.render()
    env.step(env.action_space.sample()) # take a random action