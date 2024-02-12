from math import e
import numpy as np
import torch
import time
import gymnasium.spaces as spaces
import RLLib.Agents.SAC.Core as core
'''
from spinup.utils.logx import EpochLogger
'''

class SACAgent:
    def __init__(self, env_fn, hidden_sizes=[256,256], seed=None, 
        steps_per_epoch=40000, epochs=200, replay_size=int(1e6), gamma=0.99, 
        polyak=0.9995, lr=1e-4, temp_init=0.1, temp_min=.2, batch_size=100, start_exploration_steps=5000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=50, use_HER=True, HER_ach_goal=0,
        HER_strat=core.GoalUpdateStrategy.FINAL, logger_kwargs=dict(), save_freq=1, log_max=2, log_min=-10):

        #Check for CUDA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        #Set up self
        self.env, self.test_env = env_fn, env_fn
        
        #Set HER Usage
        self.use_HER = use_HER
        self.HER_ach_goal = HER_ach_goal
        self.HER_strat = HER_strat
        
        #Set min/max entropy log std vals.
        self.log_max = log_max
        self.log_min = log_min

        #Configure obs, act, act_range, and goal dims if required for HER
        #See https://robotics.farama.org/envs/fetch/ for information regarding expected dims and types.
        self.obs_dim = np.array(self.env.observation_space['observation'].shape) if isinstance(self.env.observation_space, spaces.dict.Dict) else np.array(self.env.observation_space.shape)
        self.act_dim =  np.array(self.env.action_space.shape)
        self.act_range = [torch.tensor(self.env.action_space.low, dtype=torch.float32, device=self.device),
                          torch.tensor(self.env.action_space.high, dtype=torch.float32, device=self.device)]
        if self.use_HER:
            self.goal_dim = np.array(self.env.observation_space['desired_goal'].shape)
            self.net_obs_dim = self.goal_dim + self.obs_dim
        else:
            self.net_obs_dim = self.obs_dim

        #Use seed if one was provided.
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.seed = seed

        # Create actor critic networks and freeze targets.
        self.ac = core.MLPActorCritic(self.net_obs_dim, self.act_dim, self.act_range, hidden_sizes, log_max=self.log_max, log_min=self.log_min)
        self.ac.to(self.device)

        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.lr = lr
        self.temp_min = temp_min
        self.batch_size = batch_size
        self.start_exploration_steps = start_exploration_steps
        self.update_after = update_after
        self.update_every = update_every
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq

        #Temp tuning setup
        self.log_temp = torch.tensor(np.log(temp_init)).to(self.device)
        self.log_temp.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -self.act_dim.shape[0]


        #Optim for trained networks actor, critic1, and critic2
        self.pi_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.ac.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.ac.q2.parameters(), lr=lr)
        self.log_temp_optimizer = torch.optim.Adam([self.log_temp], lr=lr)

        #Experience replay buffer
        if self.use_HER:
            self.replay_buffer = core.HERReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, goal_dim=self.goal_dim, size=replay_size,
                                                      device=self.device, strat=self.HER_strat, HER_ach_goal=self.HER_ach_goal, k=4)
        else:
            self.replay_buffer = core.SACReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size, device=self.device)
       
        '''
        #Do something about logging, use Tensorboard here.
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())
        var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
        logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
        
        # Set up model saving
        logger.setup_pytorch_saver(ac)
        '''
        
    @property
    def temp(self):
        return self.log_temp.exp()

    def compute_loss_q(self, data):
        o, a, r, o_next, done = data['obs'], data['act'], data['rew'], data['o_next'], data['done']

        expected_q1 = self.ac.q1(o,a)
        expected_q2 = self.ac.q2(o,a)

        #Compute MBSE
        with torch.no_grad():
            #Get Target/Next action from Target/Next state/observation
            aNext, logp_aNext = self.ac.pi(o_next)

            #Compute Target Q-values and take the minimum
            q1_pi_targ = self.ac.q1targ(o_next, aNext)
            q2_pi_targ = self.ac.q2targ(o_next, aNext)
            q_pi_targ_min = torch.min(q1_pi_targ, q2_pi_targ)
            
            #Reward, Discount, Done, Target, Negative Entropy 
            backup = r + self.gamma * (1 - done) * (q_pi_targ_min - self.temp.detach() * logp_aNext)

        # MSE loss against Bellman backup
        loss_q1 = ((expected_q1 - backup)**2).mean()
        loss_q2 = ((expected_q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        '''How can I use tensorboarad here?
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())
                      
        return loss_q, q_info
        logger.store(LossQ=loss_q.item(), **q_info)
        '''
        return loss_q

    #Function to return the negative value plus entropy of the actor policy.
    def compute_negative_value_pi(self, data):
        o = data['obs']
        pi, logprob_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        #Negative entropy-regularized actor value.
        #-1(q_pi - (temp * logprob_pi)) = (-q_pi + (temp * logprob_pi)) = ((temp * logprob_pi) - q_pi).mean()
        loss_pi = (self.temp.detach() * logprob_pi - q_pi).mean()

        '''
        Log with tensorboard here.
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info
        logger.store(LossPi=loss_pi.item(), **pi_info)    
        '''
        return loss_pi, logprob_pi

    def update(self, data):
        #GD for Q1 and Q2
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        #Compute loss and grad tape graph.
        loss_q = self.compute_loss_q(data)
        #Get partials for Q1 and Q2
        loss_q.backward()
        #Take GD steps for Q1 and Q2
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        
        #Freeze the Q networks, we already optimized them.
        for q in (self.ac.q1, self.ac.q2):
            core.freeze_thaw_parameters(q)

        #Now, one GA (ASCENT FOR VALUE) step for the actor.
        self.pi_optimizer.zero_grad()
        #Compute negative actor value for negative gradient descent.
        loss_pi, logprob_pi = self.compute_negative_value_pi(data)
        #Get partials
        loss_pi.backward()
        #Take GA step for actor net pi
        self.pi_optimizer.step()

#       alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        temp_loss = (self.temp * (-logprob_pi - self.target_entropy).detach()).mean()
        self.log_temp_optimizer.zero_grad()
        temp_loss.backward()
        self.log_temp_optimizer.step()

        #Unfreeze the Q Networks
        for q in (self.ac.q1, self.ac.q2):
            core.freeze_thaw_parameters(q, freeze=False)
        
        #Lastly, for the soft part of SAC, Polyak average the Target Q networks.        
        with torch.no_grad():
            for q, qt in [(self.ac.q1, self.ac.q1targ),(self.ac.q2, self.ac.q2targ)]:
                for p, p_targ in zip(q.parameters(), qt.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)
                

    def get_action(self, o, deterministic=False, scale_action=False):
        shaped = torch.as_tensor(o.reshape(1, *self.net_obs_dim), dtype=torch.float32, device=self.device)
        action = self.ac.act(shaped, deterministic, scale_action).reshape(*self.act_dim)
        return action

    # def decay_temperature(self, epoch):
    #     #Compute annealed temperature using a linear decay schedule
    #     annealed_temp = max((self.temp_init * np.exp(-self.temp_decay*epoch)),self.temp_min)
    #     return annealed_temp

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, terminated, truncated, ep_ret, ep_len = self.test_env.reset(), False, False, 0, 0
            while not(terminated or truncated or (ep_len == self.max_ep_len)):
                o = o['observation'] if isinstance(o, dict) else o
                o, r, terminated, truncated = self.test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            
            '''Use Tensorboard here
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            '''
            
    def train(self):
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        #Vars for epoch handling, experience, logging
        obs, info, ep_ret, ep_len = *self.env.reset(), 0, 0
        o, a, r, o_next, done, dg, ag = [], [], 0, [], 0, [], []
        o = obs['observation'] if isinstance(obs, dict) else obs
            
        for t in range(total_steps):
        
            #We start off randomly exploring the environment until
            #we pass the number of initial exploration steps 
            if t < self.start_exploration_steps:
                a = self.env.action_space.sample()
            else:
                if self.use_HER:
                    cato = np.concatenate((o,dg),0)
                    a = self.get_action(cato)
                else:
                    a = self.get_action(o)

            #Perform action in environment
            obsNext, r, terminated, truncated, info = self.env.step(a)
            
            if isinstance(obsNext, dict):
                o_next = obsNext['observation']
                if self.use_HER:
                    dg, ag = obsNext['desired_goal'], obsNext['achieved_goal']
            else:
                o_next = obsNext

            #Update episode return and length                
            ep_ret += r
            ep_len += 1

            #We only want to be done if we hit the goal
            done = False if (ep_len==self.max_ep_len or truncated) else terminated

            #Send experience to replay buffer
            if self.use_HER:
                self.replay_buffer.store(o, a, r, o_next, done, dg, ag)
            else:
                self.replay_buffer.store(o, a, r, o_next, done)

            #Assign next observation to current
            o = o_next

            #End of episode handling
            if terminated or truncated or (ep_len == self.max_ep_len):
                '''Consider Tensorboard here.
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                '''
                if self.use_HER:
                   self.replay_buffer.run_goal_update_strategy(ep_len)
                
                obs, info, ep_ret, ep_len = *self.env.reset(), 0, 0
                o = obs['observation'] if isinstance(obs, dict) else o

            #Update if past initial step threshold and on an update_every multiple
            if t >= self.update_after and t % self.update_every == 0:
                for j in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(data=batch)

            #End of epoch handling
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch
                #self.temp = self.decay_temperature(epoch)

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    pass
                    '''Add Saving functionality here.
                    logger.save_state({'env': env}, None)
                    '''

                # Test the performance of the deterministic version of the agent.
                #self.test_agent()

                '''Use Tensoroboard here and log epoch info
                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('EpRet', with_min_and_max=True)
                logger.log_tabular('TestEpRet', with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)
                logger.log_tabular('TestEpLen', average_only=True)
                logger.log_tabular('TotalEnvInteracts', t)
                logger.log_tabular('Q1Vals', with_min_and_max=True)
                logger.log_tabular('Q2Vals', with_min_and_max=True)
                logger.log_tabular('LogPi', with_min_and_max=True)
                logger.log_tabular('LossPi', average_only=True)
                logger.log_tabular('LossQ', average_only=True)
                logger.log_tabular('Time', time.time()-start_time)
                logger.dump_tabular()
                '''
    def save_params(self):
        pass
   
    def load_params(self):
        pass  