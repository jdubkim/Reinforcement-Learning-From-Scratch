import gym

import setting
import config
import dqn2015

setting.set_env()
env = gym.make('SuperMarioBros-1-1-v0')
env = gym.wrappers.Monitor(env, directory="gym-results/", force=True)
#env.reset()

qNet = dqn2015.DQN2015(env)
qNet.run()
