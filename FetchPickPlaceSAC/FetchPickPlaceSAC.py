import gymnasium as gym
import math
import numpy as np
from RLLib.Agents.SAC.Agent import SACAgent

Agent = None
env_str = 'Acrobot-v1'
hidden=[512,512]
epoch_count=100
epoch_steps=25000
ep_len=250
w_HER = False
HER_obs_proc = lambda obs: None
HER_rew_function = lambda exp :0
train_agent = True
test_agent = True
log_agent = True
test_every = 5

# env_str = 'FetchPickAndPlace-v2'
# hidden=[512,512]
# epoch_count=200
# epoch_steps=5000
# ep_len=50
# w_HER = True
# HER_obs_proc = lambda obs: None
# HER_rew_function = lambda exp :0
# train_agent = True
# test_agent = True
# log_agent = True
# test_every = 10

#Adroit Hand Params
# env_str = 'AdroitHandDoorSparse-v1'
# hidden=[256,512,512,256]
# epoch_count=200
# epoch_steps=20000
# ep_len=200
# w_HER = True
# HER_obs_proc = lambda obs : (np.array([(math.pi/2)]),np.array([obs[28]]))
# HER_rew_function = lambda exp : 5
# train_agent = True
# test_agent = True
# log_agent = True
# test_every = 10

env = gym.make(env_str, max_episode_steps=ep_len)

if train_agent:
    Agent = SACAgent(env, 
                     hidden_sizes=hidden,
                     epochs=epoch_count,
                     steps_per_epoch=epoch_steps,
                     max_ep_len = ep_len,
                     use_HER= w_HER,
                     HER_obs_pr = HER_obs_proc,
                     HER_rew_func = HER_rew_function,
                     run_tests_and_record = test_agent, 
                     enable_logging = log_agent,
                     test_every_epochs = test_every)
    Agent.train()

env.close()