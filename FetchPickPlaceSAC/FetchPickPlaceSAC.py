#TODO: Turn this into a main function with **kwargs
import gymnasium as gym
import os
from RLLib.Agents.SAC.Agent import SACAgent
import RLLib.Agents.SAC.Core as core
import RLLib.Util.Functions as util

env_str_dense = 'FetchPickAndPlaceDense-v2'
env_str_sparse = 'FetchPickAndPlace-v2'

load_agent = False
train_agent = True
agent_file_path = os.path.join(os.getcwd(), 'None', 'SavedModel')
agent_file_name = 'FetchPickAndPlace-v2'
ep_len = 50
Agent = None

#Create a fetch pick and place environment.
#env = gym.make(env_str_sparse, render_mode="human", max_episode_steps=ep_len)
env = gym.make(env_str_sparse, max_episode_steps=ep_len)

#Load saved agent if we want to.
if load_agent:
    Agent = util.load(agent_file_name, agent_file_path)
    #Agent.env = env
    #util.setup_test_env(Agent, 'TestRecordings')
    #Agent.configure_buffer()

#Configure and train SACAgent.
if train_agent:
    Agent = SACAgent(env, max_ep_len = ep_len, HER_strat=core.GoalUpdateStrategy.FUTURE, run_tests_and_record=True)
    Agent.train()

env.close()