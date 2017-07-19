# -*- coding: utf-8 -*-
import sys
import time

import numpy as np
import cv2

from constants import GYM_MONITOR_DIR
from constants import ACTION_SPACE_TYPE

import gym
import gym.utils
from gym import wrappers
import gym_doom
from gym_doom.wrappers import *

class GameState(object):
    def __init__(self, rand_seed, ROM, display=True, no_op_max=7, task_index=-1):
        self.task_index = task_index
        self.ROM = ROM
        action_space_wrapper = ToDiscrete(ACTION_SPACE_TYPE)
        self.env = gym.make(self.ROM)
        self.env.close()
        self.env = action_space_wrapper(self.env)
        self.display = display
        if (self.display):
            self.env = wrappers.Monitor(self.env, GYM_MONITOR_DIR + '-' + self.ROM)
        self.env.seed(rand_seed)
        self._no_op_max = no_op_max
        # self.real_actions = self.env.action_space
        self._screen = np.empty((480, 640, 1), dtype=np.uint8)
        self.reset()

    def _process_frame(self, action, reshape):
        observation, reward, terminal, info = self.env.step(action)
        self._screen = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        reshaped_screen = np.reshape(self._screen, (480, 640))

        resized_screen = cv2.resize(reshaped_screen, (120, 160))
        x_t = resized_screen[:,10:]
        if reshape:
            x_t = np.reshape(x_t, (160, 110, 1))
        x_t = x_t.astype(np.float32)
        x_t *= (1.0/255.0)
        return reward, terminal, x_t

    def reset(self):
        self.env.reset()
        time.sleep(3)
        # randomize initial state
        if self._no_op_max > 0:
            no_op = np.random.randint(0, self._no_op_max + 1)
            for _ in range(no_op):
                self.env.step(0)

        _, _, x_t = self._process_frame(0, False)

        self.reward = 0
        self.terminal = False
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    def process(self, action):
        # if (action > self._no_op_max - 1):
        #     # print("Action '{}' is out of bounds. Remapped action '{}' -> '{}'".format(action, action, self._no_op_max-1))
        #     action = self._no_op_max - 1

        r, t, x_t1 = self._process_frame(action, True)
        self.reward = r
        self.terminal = t
        self.s_t1 = np.append(self.s_t[:,:,1:], x_t1, axis = 2)

    def update(self):
        self.s_t = self.s_t1

    def close_env(self):
        self.env.close()
