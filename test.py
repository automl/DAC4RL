from rlenv.RLEnv import RLEnv

from carl.envs import* 
import numpy as np 
import gym



rle = gym.make("rl-v0", env=CARLPendulumEnv)

print(f'Ive got the magic')
