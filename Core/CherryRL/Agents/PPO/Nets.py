from abc import ABC
import numpy as np
from gymnasium.spaces import Box, Discrete
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import CherryRL.Util.Functions as funcs

#Shared actor components for derived classes, abstract base class.
class Actor(nn.Module, ABC):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        #Produce action distributions for a given set of observations, and 
        #calculate the log likelihood for selected actions within produced distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

#Actor net for discrete action spaces.
class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = funcs.mlp(list(obs_dim) + list(hidden_sizes) + list(act_dim), activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

#Actor net for continuous action spaces.
class MLPGaussianActor(Actor):
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

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self._mu = 0
        self._std = 0
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = funcs.mlp(list(obs_dim) + list(hidden_sizes) + list(act_dim), activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        self.mu = mu
        self.std = std
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.value_net = funcs.mlp(list(obs_dim) + list(hidden_sizes) + list([1]), activation)

    def forward(self, obs):
        return torch.squeeze(self.value_net(obs), -1)

class MLPActorCritic(nn.Module):
    def __init__(self, agent, hidden_sizes=[256,256], activation=nn.Tanh):
        super().__init__()
        self.device = agent.device

        if agent.action_discrete:
            self.pi = MLPCategoricalActor(agent.net_obs_dim, agent.num_discrete_actions, hidden_sizes, activation)
        else:
            self.pi = MLPGaussianActor(agent.net_obs_dim, agent.act_dim, hidden_sizes, activation)

        self.value  = MLPCritic(agent.net_obs_dim, hidden_sizes, activation)

    def step(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            action = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, action)
            value = self.value(obs)
        return action.cpu().numpy(), value.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]