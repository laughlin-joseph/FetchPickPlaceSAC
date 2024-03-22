import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import CherryRL.Util.Functions as funcs

class SquashedGaussianMLPActor(nn.Module):
    @property
    def mu(self):
        if self._mu is None:
            return 0
        else:
            if isinstance(self._mu, torch.Tensor):
                return self._mu.squeeze() 
            else: 
                return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value
    
    @property
    def std(self):
        if self._std is None:
            return 0
        else:
            if isinstance(self._std, torch.Tensor):
                return self._std.squeeze() 
            else: 
                return self._std

    @std.setter
    def std(self, value):
        self._std = value

    @property
    def entropy(self):
        if self._entropy is None:
            return 0
        else:
            return self._entropy

    @entropy.setter
    def entropy(self, value):
        self._entropy = value

    def __init__(self, obs_dim, act_dim, hidden_sizes, discrete, num_dis_actions, activation, log_max, log_min):
        super().__init__()
        self.discrete = discrete
        self.log_max = log_max
        self.log_min = log_min
        self.mu = 0
        self.std = 0
        self.entropy = 0
        self.num_dis_actions = num_dis_actions
        self.obs_dim = obs_dim
        self.act_dim = num_dis_actions if discrete else act_dim[0]

        if discrete:
            self.net = funcs.mlp(list(obs_dim) + list(hidden_sizes) + list(self.act_dim), activation, None)
            self.soft_max = nn.Softmax(-1)
        else:
            self.net = funcs.mlp(list(obs_dim) + list(hidden_sizes), activation, activation)
            self.mu_layer = nn.Linear(hidden_sizes[-1], self.act_dim)
            self.log_std_layer = nn.Linear(hidden_sizes[-1], self.act_dim)

    def forward(self, obs, deterministic=True, with_logprob=True):
        #For discrete action spaces.
        if self.discrete:
            raw_probs = self.net(obs)
            pi_probs = self.soft_max(raw_probs)
            #https://pytorch.org/docs/stable/distributions.html#categorical
            pi_distribution = Categorical(pi_probs)
            if deterministic:
                pi_action = torch.argmax(pi_probs, axis=-1)
            else:
                pi_action = pi_distribution.sample()
            #Where pi_probs is zero, replace with a very small number, ln(0) is undefined.
            eps = pi_probs == 0.0
            eps = eps.float() * 1e-8
            pi_probs = pi_probs + eps
            logprob_pi = torch.log(pi_probs)
            probs = (logprob_pi, pi_probs)
            self.entropy = pi_probs * logprob_pi
        
        #For continuous action spaces.
        else:
            net_out = self.net(obs)
            mu = self.mu_layer(net_out)
            log_std = self.log_std_layer(net_out)
            log_std = torch.clamp(log_std, self.log_min, self.log_max)
            std = torch.exp(log_std)
            
            self.mu = mu.detach()
            self.std = std.detach()
            #https://pytorch.org/docs/stable/distributions.html#normal
            pi_distribution = Normal(mu, std)
            if deterministic:
                #Average returned when testing
                pi_action = mu
            else:
                #Sample reparameterized action 
                pi_action = pi_distribution.rsample()

            if with_logprob:
                #Original equation from paper, section C eqn 21.
                #logprob_pi = ((pi_distribution.log_prob(pi_action).sum(dim=-1) - torch.log(1 - torch.tanh(pi_action).pow(2)))).sum(dim=-1)
                #Using an equation that is more numerically stable:
                #https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
                logprob_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
                logprob_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
                self.entropy = logprob_pi
            else:
                logprob_pi = None
            
            probs = (logprob_pi)
            pi_action = torch.tanh(pi_action)

        return pi_action, probs


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, discrete, num_dis_actions, activation):
        super().__init__()
        self.discrete = discrete
        self.num_dis_actions = num_dis_actions
        self.obs_dim = obs_dim
        self.act_dim = num_dis_actions if discrete else act_dim[0]
        if discrete:
            self.q = funcs.mlp(list(obs_dim) + list(hidden_sizes) + list(num_dis_actions), activation)            
        else:
            #Cat obs and act dims for input layer, add hidden layers, add output Q.
            self.q = funcs.mlp(list(obs_dim + act_dim) + list(hidden_sizes) + list([1]), activation)

    def forward(self, input = list):
        obs = input[0]
        if self.discrete:
            q = self.q(obs)
        else:
            #Output a 
            act = input[1]
            q = self.q(torch.cat([obs, act], axis=-1))
            q = q.squeeze(-1)
        
        return q

class MLPActorCritic(nn.Module):
    def __init__(self, agent, hidden_sizes=[256,256], activation=nn.ReLU):
        super().__init__()
        self.device = agent.device

        #Build actor, critic1, critic2, targ1, targ2 networks.
        self.pi = SquashedGaussianMLPActor(agent.net_obs_dim, agent.act_dim, hidden_sizes, agent.action_discrete, 
                                           agent.num_discrete_actions, activation, agent.log_max, agent.log_min)
        self.q1 = MLPQFunction(agent.net_obs_dim, agent.act_dim, hidden_sizes, agent.action_discrete, agent.num_discrete_actions, activation)
        self.q2 = MLPQFunction(agent.net_obs_dim, agent.act_dim, hidden_sizes, agent.action_discrete, agent.num_discrete_actions, activation)
        self.q1targ = MLPQFunction(agent.net_obs_dim, agent.act_dim, hidden_sizes, agent.action_discrete, agent.num_discrete_actions, activation)
        self.q2targ = MLPQFunction(agent.net_obs_dim, agent.act_dim, hidden_sizes, agent.action_discrete, agent.num_discrete_actions, activation)
        
        #Freeze target networks, these are updated with a Polyak average.
        funcs.freeze_thaw_parameters(self.q1targ)
        funcs.freeze_thaw_parameters(self.q2targ)
    
    def act(self, obs, deterministic=False):
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs.reshape(1, *self.pi.obs_dim), dtype=torch.float32, device=self.device)
            a, _ = self.pi(obs, deterministic, with_logprob=False)
            a = a.cpu().numpy() if isinstance(a, torch.Tensor) else a
            if self.pi.discrete:
                a = int(a)
            else:
                a = a.reshape(self.pi.act_dim)
            return a