# -*- coding: utf-8 -*-
import time

LOCAL_T_MAX = 20 # repeat step size
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
CHECKPOINT_DIR = 'checkpoints'
INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate

PARALLEL_SIZE = 4 # parallel thread size
# ROMZ = ["Alien-v0", "Centipede-v0"]
# ACTION_SIZEZ= [18, 18]
ROMZ = ["gym_doom/DoomBasic-v0", "gym_doom/DoomDeathmatch-v0"]
ACTION_SPACE_TYPE = 'constant-7'
ACTION_SIZEZ= [8, 8] # using 'constant-7' -> Discrete action_space configuration

GYM_MONITOR_DIR = "./data/gym/experiment-" + str(int(time.time()))  # directory for Gym monitor to log

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 0.01 # entropy regurarlization constant
MAX_TIME_STEP = 1 * 10**5
GRAD_NORM_CLIP = 40.0 # gradient norm clipping
NUM_GPUS = 3
USE_GPU = False # To use GPU, set True
USE_LSTM = False # True for A3C LSTM, False for A3C FF
USE_PATHNET = True # True for A3C PathNet, False for A3C FF
