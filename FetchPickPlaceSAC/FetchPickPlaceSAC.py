#TODO: Turn this into a main function with **kwargs
import gymnasium as gym
import math
import numpy as np
from RLLib.Agents.SAC.Agent import SACAgent

Agent = None
#env_str = 'FetchPickAndPlace-v2'
env_str = 'AdroitHandDoorSparse-v1'
ep_len = 200
w_HER = True
HER_obs_proc = lambda obs : (np.array([(math.pi/2)]),np.array([obs[28]]))
HER_rew_function = lambda exp : 5
train_agent = True
test_agent = True
log_agent = True

env = gym.make(env_str, max_episode_steps=ep_len)

if train_agent:
    Agent = SACAgent(env, 
                     max_ep_len = ep_len,
                     use_HER= w_HER,
                     HER_obs_pr = HER_obs_proc,
                     HER_rew_func = HER_rew_function,
                     run_tests_and_record = test_agent, 
                     enable_logging = log_agent)
    Agent.train()

env.close()