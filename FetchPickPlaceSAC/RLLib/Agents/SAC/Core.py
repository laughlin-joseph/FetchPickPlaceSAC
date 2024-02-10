from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def freeze_thaw_parameters(module, freeze=True):
    if freeze:
        for p in module.parameters():
            p.requires_grad = False
    else:
        for p in module.parameters():
            p.requires_grad = True
            
class SACReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.o_next_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        
        self.ptr, self.size, self.max_size, self.device = -1, 0, size, device

    def store(self, obs, act, rew, obs_next, done):
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.o_next_buf[self.ptr] = obs_next
        self.done_buf[self.ptr] = done

    def sample_batch(self, batch_size=50):
        indexes = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[indexes],
                     act=self.act_buf[indexes],
                     rew=self.rew_buf[indexes],
                     o_next=self.o_next_buf[indexes],
                     done=self.done_buf[indexes])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in batch.items()}

class GoalUpdateStrategy(Enum):
    FINAL = 1
    FUTURE = 2
    EPISODE = 3

#HER buffer, see https://arxiv.org/pdf/1707.01495.pdf
#k is number of virtual copies per replayed step.
class HERReplayBuffer(SACReplayBuffer):
    def __init__(self, obs_dim, act_dim, goal_dim, size, device, strat=GoalUpdateStrategy.FINAL, HER_ach_goal=0, k=4):
        super().__init__(obs_dim, act_dim, size, device)
        self.desired_goal_buf = np.zeros(combined_shape(size, goal_dim), dtype=np.float32)
        self.achieved_goal_buf = np.zeros(combined_shape(size, goal_dim), dtype=np.float32)
        self.strat = strat
        self.k = k
        self.HER_ach_goal = HER_ach_goal
    
    def store(self, obs, act, rew, obs_next, done, desired_goal, achieved_goal):
        SACReplayBuffer.store(self, obs, act, rew, obs_next, done)
        # Store achieved goal and desired goal along with the transition
        self.desired_goal_buf[self.ptr] = desired_goal
        self.achieved_goal_buf[self.ptr] = achieved_goal

    def sample_batch(self, batch_size=50):
        indexes = np.random.randint(0, self.size, size=batch_size)
        desired_goal = self.desired_goal_buf[indexes]
        batch = dict(obs=np.concatenate((self.obs_buf[indexes], desired_goal), axis=1),
                     act=self.act_buf[indexes],
                     rew=self.rew_buf[indexes],
                     o_next=np.concatenate((self.o_next_buf[indexes], desired_goal), axis=1),
                     done=self.done_buf[indexes])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in batch.items()}
    
    def run_goal_update_strategy(self, batch_size):
        start, end, cur = (self.ptr-(batch_size-1)), self.ptr, 0
        process_list = []
        
        for i in range(start, end+1):
            process_list.append({'obs':self.obs_buf[i],
                                'act':self.act_buf[i],
                                'rew':self.rew_buf[i],
                                'o_next':self.o_next_buf[i],
                                'done':self.done_buf[i],
                                'des':self.desired_goal_buf[i],
                                'ach':self.achieved_goal_buf[i]})

        #TODO:Add strategies and take max reward as Agent parameter
        batch_last = batch_size - 1
        sample_end = batch_last - self.k
        final = process_list[batch_last]['ach']
        for pos, exp in enumerate(process_list):
            match self.strat:
                case GoalUpdateStrategy.FINAL:
                    self.store(exp['obs'],
                               exp['act'],
                               self.HER_ach_goal,
                               exp['o_next'],
                               exp['done'],
                               final,exp['ach'])
                    
                case GoalUpdateStrategy.FUTURE:
                    if pos < (sample_end):
                        future_goal = exp['ach']
                        virtIndexes = np.random.randint(pos+1, high=sample_end+1, size=self.k)
                        for idx in virtIndexes:
                            self.store(process_list[idx]['obs'],
                                       process_list[idx]['act'],
                                       self.HER_ach_goal,
                                       process_list[idx]['o_next'],
                                       process_list[idx]['done'],
                                       future_goal,
                                       process_list[idx]['ach'])
                
                case GoalUpdateStrategy.EPISODE:
                    ep_goal = exp['ach']
                    virtIndexes = np.random.choice(range(0, pos) + range(pos+1, high=batch_last), size=self.k)
                    for idx in virtIndexes:
                        self.store(process_list[idx]['obs'],
                                    process_list[idx]['act'],
                                    self.HER_ach_goal,
                                    process_list[idx]['o_next'],
                                    process_list[idx]['done'],
                                    ep_goal,
                                    process_list[idx]['ach'])

class SquashedGaussianMLPActor(nn.Module):
    
    def __init__(self, obs_dim, act_dim, act_range, hidden_sizes, activation, ent_max, ent_min):
        super().__init__()
        self.ent_max = ent_max
        self.ent_min = ent_min
        self.net = mlp(list(obs_dim) + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim[0])
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim[0])
        self.act_min, self.act_max = act_range[0], act_range[1]

    def forward(self, obs, deterministic=False, with_logprob=True, scale_tanh=False):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.ent_min, self.ent_max)
        std = torch.exp(log_std)

        #Sample reparameterized action 
        pi_distribution = Normal(mu, std)
        if deterministic:
            #Average returned when testing
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        pi_action = torch.tanh(pi_action)

        if with_logprob:
            #Small constant added to avoid ln(0)
            logprob_pi = (pi_distribution.log_prob(pi_action) - torch.log(1 - pi_action.pow(2) + 1e-6)).sum(dim=1)
        else:
            logprob_pi = None
        
        if scale_tanh:
            #Original is in [omin, omax], target is in [tmin,tmax]
            #Target = ((original - rmin)/(rmax - rmin)) * (tmax - tmin) + tmin
            squish_min, squish_max = -1, 1
            #Scale to action range.
            pi_action = ((pi_action - squish_min)/(squish_max - squish_min)) * (self.act_max - self.act_min) + self.act_min
        
        return pi_action, logprob_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        
        #Cat obs and act dims for input layer, add hidden layers, add output Q.
        self.q = mlp(list(obs_dim + act_dim) + list(hidden_sizes) + list([1]), activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        #Reshape val returned from Q network MLP.
        return torch.squeeze(q, -1)

class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, act_range, hidden_sizes=[256,256], activation=nn.ReLU, ent_max=2, ent_min=-20):
        super().__init__()

        #Build actor, critic1, critic2, targ1, targ2 networksS
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, act_range, hidden_sizes, activation, ent_max, ent_min)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q1targ = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2targ = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        
        #Freeze target networks, these are updated with a Polyak average.
        freeze_thaw_parameters(self.q1targ)
        freeze_thaw_parameters(self.q2targ)

    def act(self, obs, deterministic=False, scale_action=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, with_logprob=False, scale_tanh=scale_action)
            a = a.cpu()
            return a.numpy()