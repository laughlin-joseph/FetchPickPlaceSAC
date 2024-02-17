import copy
from datetime import datetime
import dill as pickle
import gymnasium as gym
import numpy as np
import os
import random
import torch
import torch.nn as nn

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def freeze_thaw_parameters(module, freeze=True):
    if freeze:
        for p in module.parameters():
            p.requires_grad = False
    else:
        for p in module.parameters():
            p.requires_grad = True

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def set_dirs(agent, folder='SavedModel'):
    now = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    agent.root_dir = os.path.join(os.getcwd(), now)
    agent.save_dir = os.path.join(agent.root_dir, folder)
            
def load(filename, folder):
    load_dir = os.path.join(os.getcwd(), folder)
    with open(os.path.join(load_dir, filename), 'rb') as f:
        agent = pickle.load(f)
    return agent

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def save(agent, filename):
    agent_copy = copy.deepcopy(agent)
    agent_copy.env = None
    agent_copy.test_env = None
    agent_copy.replay_buffer = None
    if not agent_copy.save_dir:
        set_dirs(agent_copy)
    if not os.path.exists(agent_copy.save_dir):
        os.mkdir(agent_copy.save_dir)
    with open(os.path.join(agent_copy.save_dir, filename), 'wb+') as f:
        pickle.dump(agent_copy, f)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def setup_test_env(agent, rec_folder):
    if not agent.env or not agent.save_dir:
       raise ValueError("Agent.env and Agent.save_dir must be configured.")
    
    rec_folder = os.path.join(agent.root_dir, rec_folder)
    max_len = agent.max_ep_len if agent.max_ep_len else 0
    
    agent.test_env = gym.make(agent.env_name, render_mode='rgb_array', max_episode_steps=agent.max_ep_len)
    
    agent.test_env = gym.wrappers.RecordVideo(env=agent.test_env, video_folder=rec_folder, 
                                    episode_trigger=lambda a : True, video_length=max_len, 
                                    name_prefix=agent.test_count)
    
    max_len = max_len if max_len > 0 else 100
    agent.test_env = gym.wrappers.RecordEpisodeStatistics(agent.test_env, deque_size=max_len)