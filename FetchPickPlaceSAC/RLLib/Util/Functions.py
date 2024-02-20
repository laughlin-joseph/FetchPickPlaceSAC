from datetime import datetime
import dill as pickle
import gymnasium as gym
import numpy as np
import os
import random
from tensorboard import program
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def create_summary_writer(agent, log_dir_name='logdata'):
    if not hasattr(agent, 'root_dir') or not agent.root_dir:
       set_dirs(agent)
    agent.log_data_dir = os.path.join(agent.root_dir, log_dir_name)
    agent.writer = SummaryWriter(agent.log_data_dir)

def freeze_thaw_parameters(module, freeze=True):
    if freeze:
        for p in module.parameters():
            p.requires_grad = False
    else:
        for p in module.parameters():
            p.requires_grad = True

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
            
def load(filename, folder):
    load_dir = os.path.join(os.getcwd(), folder)
    with open(os.path.join(load_dir, filename), 'rb') as f:
        agent = pickle.load(f)
    return agent

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    act = ''
    for j in range(len(sizes)-1):
        ll = nn.Linear(sizes[j], sizes[j+1])
        nn.init.kaiming_uniform_(ll.weight, nonlinearity='relu')
        layers.append(ll)
        act = activation if j < len(sizes)-2 else output_activation
        layers.append(act())
    return nn.Sequential(*layers)

def save(agent, filename):
    if not hasattr(agent, 'root_dir') or not agent.root_dir:
        set_dirs(agent)
    if not os.path.exists(agent.save_dir):
        os.mkdir(agent.save_dir)
    with open(os.path.join(agent.save_dir, filename), 'wb+') as f:
        pickle.dump(agent, f)

def set_dirs(agent, folder='SavedModel'):
    if not hasattr(agent, 'root_dir') or not agent.root_dir or not agent.save_dir:
        now = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        agent.root_dir = os.path.join(os.getcwd(), now)
        agent.save_dir = os.path.join(agent.root_dir, folder)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def setup_test_env(agent, rec_folder):
    if not agent.root_dir:
       set_dirs(agent)
    
    rec_folder = os.path.join(agent.root_dir, rec_folder)
    max_len = agent.max_ep_len if agent.max_ep_len else 0
    
    agent.test_env = gym.make(agent.env_name, render_mode='rgb_array', max_episode_steps=agent.max_ep_len)
    
    agent.test_env = gym.wrappers.RecordVideo(env=agent.test_env, video_folder=rec_folder, 
                                    episode_trigger=lambda a : True, video_length=max_len, 
                                    name_prefix=agent.test_count)
    
    max_len = max_len if max_len > 0 else 100
    agent.test_env = gym.wrappers.RecordEpisodeStatistics(agent.test_env, deque_size=max_len)

def start_tensorboard(logdir):
    running, board_add = False, ''
    try:
        board = program.TensorBoard()
        board.configure(argv=[None, '--logdir', logdir])
        board_add = board.launch()
        print(f"Tensorboard running at: {board_add}")
        running = True
    except Exception as e:
        print("Error starting Tensorboard: ", e)
    
    return running, board_add

def find_oldest_file(directory, file_extension):
    return min((os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(file_extension)),
               key=os.path.getmtime, default=None)
    