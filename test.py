import gym
import gym.utils
from gym import wrappers
import gym_doom
from gym_doom.wrappers import *
import time


wrapper = ToDiscrete('constant-7')
env = wrapper(gym.make('gym_doom/DoomBasic-v0'))
env.close()
env.reset()

for a in range(8):
    print 'ACTION:', a
    for i in range(350):
        _, reward, _, _ = env.step(a)
        print i, reward
        env.render()
    env.reset()
env.close()
