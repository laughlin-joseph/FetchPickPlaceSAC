import cv2
from datetime import datetime
import dill as pickle
import gymnasium as gym
import numpy as np
import os
import random
import scipy.signal
from tensorboard import program
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def create_summary_writer(agent, log_dir_name='logdata', net_names=[]):
    if not hasattr(agent, 'root_dir') or not agent.root_dir:
       set_dirs(agent)
    net_writers = {}
    
    base_log_dir = os.path.join(agent.root_dir, log_dir_name)
    agent.log_data_dir = base_log_dir

    agent.writer = SummaryWriter(base_log_dir)

    if net_names:
        for name in net_names:    
            cur_path = os.path.join(base_log_dir, name)
            net_writers[name] = SummaryWriter(cur_path)
            
    return  net_writers

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def freeze_thaw_parameters(module, freeze=True):
    if freeze:
        for p in module.parameters():
            p.requires_grad = False
    else:
        for p in module.parameters():
            p.requires_grad = True

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_environment_shape(agent):
    env = agent.env
    device = get_device()
    obs_dim = 0
    act_dim = 0
    agent.action_discrete = False
    agent.num_discrete_actions = 0

    if isinstance(env.observation_space, gym.spaces.dict.Dict):
        obs_dim = np.array(env.observation_space['observation'].shape)
    elif isinstance(env.observation_space, gym.spaces.box.Box):
        obs_dim = np.array(env.observation_space.shape)
    elif isinstance(env.observation_space, gym.spaces.discrete.Discrete):
        obs_dim = np.array(1).reshape(1,)
    else:
        obs_dim = np.array(env.observation_space.shape)

    if isinstance(env.action_space, gym.spaces.dict.Dict):
        #Not sure this will ever be a thing, would be something like:
        #space = env.action_space['action']
        #act_dim = np.array(space.shape)
        raise NotImplementedError
    elif isinstance(env.action_space, gym.spaces.box.Box):
        act_dim =  np.array(env.action_space.shape)
    elif isinstance(env.action_space, gym.spaces.discrete.Discrete):
        act_dim =  np.array(1).reshape(1,)
        agent.action_discrete = True
        agent.num_discrete_actions = env.action_space.n
    else:
        act_dim =  np.array(env.action_space.shape)
    
    return obs_dim, act_dim

def get_latest_frames(directory, file_extension):
    vid_path =  max((os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(file_extension)), 
                    key=os.path.getmtime, default=None)
    vid = cv2.VideoCapture(vid_path)
    
    if not vid.isOpened():
        return None
    fps = vid.get(cv2.CAP_PROP_FPS)
    frames = []
    while vid.isOpened():
        ret, frame = vid.read()        
        if not ret:
            break
        frames.append(frame)
    vid.release()

    cv2.destroyAllWindows()
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    vid_tensor = torch.from_numpy(np.array(rgb_frames))
    vid_tensor = vid_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3)
    
    return vid_tensor, fps
            
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
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)

def sample_categorical(probs, n=100):
    dist = torch.distributions.categorical.Categorical(probs)
    probs_shape = [sv for i, sv in enumerate(probs.shape) if i > 0]
    sample_shape = combined_shape(n, probs_shape)
    vals = dist.sample(sample_shape)
    return vals

def sample_normal(mu, sigma, n=100):
    dist = torch.distributions.normal.Normal(mu, sigma)
    mu_shape = [sv for i, sv in enumerate(mu.shape) if i > 0]
    sample_shape = combined_shape(n, mu_shape)
    vals = dist.sample(sample_shape)
    return vals

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
    
    agent.test_env = gym.make(agent.env_name, render_mode='rgb_array', max_episode_steps=agent.max_ep_len)
    
    agent.test_env = gym.wrappers.RecordVideo(env=agent.test_env, video_folder=rec_folder, 
                                              episode_trigger=lambda a : True, name_prefix=agent.test_count)
    
    agent.test_env = gym.wrappers.RecordEpisodeStatistics(agent.test_env)

def start_tensorboard(logdir):
    running, board_add = False, ''
    try:
        board = program.TensorBoard()
        board.configure(argv=[None, '--logdir', logdir])
        board_add = board.launch()
        print('\nTensorboard running at: {%s}' % board_add)
        running = True
    except Exception as e:
        print('\nError starting Tensorboard: %s' % e)
    
    return running, board_add