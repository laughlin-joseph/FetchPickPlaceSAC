from math import e
import os
import pickle
import numpy as np
import torch
import time
import gymnasium.spaces as spaces
import RLLib.Agents.SAC.Core as core
import RLLib.Util.Functions as util
import RLLib.Util.Video as video
import RLLib.Util.EnvWrappers as wrappers

#40000 - default steps_per_epoch
class SACAgent:
    def __init__(self, env_fn, hidden_sizes=[512,512], seed=1, 
                 epochs=200, steps_per_epoch=5000, max_ep_len=50, save_freq=5,
                 gamma=0.95, polyak=0.9995, lr=5e-4, temp_init=1.0, log_max=2, log_min=-10,
                 batch_size=1024, replay_buffer_size=int(1e6), use_HER=True, HER_rew_func=lambda:0, HER_strat=core.GoalUpdateStrategy.FINAL, HER_k=1,
                 start_exploration_steps=5000, update_after=10000, update_every_steps=100, run_tests=False,
                 test_record_video=False, done_at_goal=False):

        #Check for CUDA.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        #Envs.
        self.env, self.test_env = env_fn, env_fn
        
        #Set HER usage and buffer props.
        self.use_HER = use_HER
        self.HER_rew_func = HER_rew_func
        self.HER_strat = HER_strat
        self.HER_k = HER_k
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        
        #Set params and constants.
        self.gamma = gamma
        self.polyak = polyak
        self.lr = lr
        self.temp_init = temp_init
        self.log_max = log_max
        self.log_min = log_min

        #Configure obs, act, act_range, and goal dims if required for HER.
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

        #Epochs and episode length
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq

        #Initial exploration and update cycle management.
        self.start_exploration_steps = start_exploration_steps
        self.update_after = update_after
        self.update_every_steps = update_every_steps
        
        #Testing
        self.run_tests = run_tests

        #Use seed if one was provided.
        util.set_seed(seed)

        #Temp tuning setup
        self.log_temp = torch.tensor(np.log(self.temp_init)).to(self.device)
        self.log_temp.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -self.act_dim[0]

        # Create actor critic networks and freeze targets.
        self.ac = core.MLPActorCritic(self.net_obs_dim, self.act_dim, self.act_range, hidden_sizes, log_max=self.log_max, log_min=self.log_min)
        self.ac.to(self.device)

        #Optim for trained networks actor, critic1, and critic2. Optim for temp dual func.
        self.pi_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.ac.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.ac.q2.parameters(), lr=lr)
        self.log_temp_optimizer = torch.optim.Adam([self.log_temp], lr=lr)

        #Experience replay buffer.
        if self.use_HER:
            self.replay_buffer = core.HERReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, goal_dim=self.goal_dim, size=self.replay_buffer_size,device=self.device,
                                                      strat=self.HER_strat, HER_rew_func=self.HER_rew_func, k=self.HER_k)
        else:
            self.replay_buffer = core.SACReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_buffer_size, device=self.device)

        #Environment Wrappers
        if test_record_video:
            #self.env = wrappers.ImageEnv(self.env)
            #Try passing env to video object and call env.render within it.
            self.video_recorder = video.VideoRecorder(self.seed)
            self.video_recorder.init()
        if done_at_goal:
            #Consider droppng env wrappers, they may not be necessary any longer.
            self.env = wrappers.DoneOnSuccessWrapper(self.env)

        #Do something about logging, use Tensorboard here.
        #logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
        
    @property
    def temp(self):
        return self.log_temp.exp()
    
    @staticmethod
    def load(filename, folder='SavedModels'):
        load_dir = os.path.join(os.getcwd(), folder)
        with open(os.path.join(load_dir, filename), 'rb') as f:
            agent = pickle.load(f)
        return agent

    def save(self, filename, folder='SavedModels'):
        self.save_dir = os.path.join(os.getcwd(), folder)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        with open(os.path.join(self.save_dir, filename), 'wb+') as f:
            pickle.dump(self, f)
   
    #Minimize the Bellman residual.
    #Section 4.2 Equation 5 of https://arxiv.org/pdf/1812.05905v2.pdf
    #Q(st, at) - (r(st, at) + gamma * V(st+1))
    #Where V(st+1)= (Q(st+1, at+1) - temp * log(pi(at+1)))
    def compute_loss_q(self, data):
        o, a, r, o_next, done = data['obs'], data['act'], data['rew'], data['o_next'], data['done']

        #TD Present target.
        expected_q1 = self.ac.q1(o,a)
        expected_q2 = self.ac.q2(o,a)

        #Compute MBSE.
        #Dont update the targets here!
        with torch.no_grad():
            #Get Target TD action from next observation.
            aNext, logp_aNext = self.ac.pi(o_next)

            #Compute Target Q-values and take the minimum.
            q1_pi_targ = self.ac.q1targ(o_next, aNext)
            q2_pi_targ = self.ac.q2targ(o_next, aNext)
            q_pi_targ_min = torch.min(q1_pi_targ, q2_pi_targ)
            
            #Reward, Discount, Done, QTarget, Added Entropy.
            backup = r + self.gamma * (1 - done) * (q_pi_targ_min - self.temp.detach() * logp_aNext)

        #MSE loss against Bellman backup, take average of Q net loss and return.
        loss_q1 = ((expected_q1 - backup)**2).mean()
        loss_q2 = ((expected_q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        #How can I use tensorboarad here?
        #q_info = dict(Q1Vals=q1.detach().numpy(),Q2Vals=q2.detach().numpy())
        #logger.store(LossQ=loss_q.item(), **q_info)

        return loss_q

    #Function to return pi loss.
    #See Section 4.2 Equations 7, 8, and 9 of https://arxiv.org/pdf/1812.05905v2.pdf
    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logprob_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Max V(st) = (Q(st, at) - temp * log(pi(at)))
        # by min (temp * log((pi(atR))) - Q(st, atR)
        # where atR is pi's reparametrized  output.
        # Pi outputs a mu and std actions are reparametrized  by
        # noise from a normal distribution then scaled and shifted
        # by pi's output, the reparameterization trick.
        loss_pi = (self.temp.detach() * logprob_pi - q_pi).mean()

        #Log with tensorboard here.
        #pi_info = dict(LogPi=logp_pi.detach().numpy())
        #logger.store(LossPi=loss_pi.item(), **pi_info)    
        
        return loss_pi, logprob_pi

    def update(self, data):
        #GD for Q1 and Q2.
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        #Compute loss and grad tape graph.
        loss_q = self.compute_loss_q(data)
        #Get partials for Q1 and Q2.
        loss_q.backward()
        #Take GD steps for Q1 and Q2.
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        
        #Freeze the Q networks, we already optimized them.
        for q in (self.ac.q1, self.ac.q2):
            util.freeze_thaw_parameters(q)

        #Optimize the policy.
        self.pi_optimizer.zero_grad()
        #Compute pi/actor loss.
        loss_pi, logprob_pi = self.compute_loss_pi(data)
        #Get partials.
        loss_pi.backward()
        #Take GD step for actor net pi.
        self.pi_optimizer.step()

        #Adjust the temp parameter. Detach logprob_pi and target_entropy from optim.
        #See section 5 equation 17 of https://arxiv.org/pdf/1812.05905v2.pdf
        temp_loss = (self.temp * (-logprob_pi - self.target_entropy).detach()).mean()
        self.log_temp_optimizer.zero_grad()
        temp_loss.backward()
        self.log_temp_optimizer.step()

        #Unfreeze the Q Networks.
        for q in (self.ac.q1, self.ac.q2):
            util.freeze_thaw_parameters(q, freeze=False)
        
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

    def test_agent(self, epoch):
        test_reward = 0
        average_episode_reward = 0
        
        obs = self.env.reset()
        o, a, r, dg= [], [], 0, []
        o = obs['observation'] if isinstance(obs, dict) else obs
                    
        for _ in range(self.max_ep_len):
            
            if self.use_HER:
                dg= o['desired_goal']
                cato = np.concatenate((o,dg),0)
                a = self.get_action(cato)
            else:
                a = self.get_action(o)
            
            obsNext, r = self.env.step(a)
            
            if self.test_record_video:
                self.video_recorder.record(obsNext)                
            
            test_reward += r
            average_episode_reward += test_reward/self.num_eval_episodes
                
            o = obsNext['observation'] if isinstance(obs, dict) else obsNext                
            if self.use_HER:
                dg = o['desired_goal']
                o = np.concatenate((o,dg),0)
        
        #Log with Tensorboard here!                

        if self.test_record_video:        
            self.video_recorder.save(f'{epoch}.mp4')
            
    def train(self):
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        #Vars for epoch handling, experience, Tensorboard.
        obs, info, ep_ret, ep_len = *self.env.reset(), 0, 0
        o, a, r, o_next, done, dg, ag = [], [], 0, [], 0, [], []
        o = obs['observation'] if isinstance(obs, dict) else obs
        if self.use_HER:
            dg, ag = o['desired_goal'], o['achieved_goal']
            
        for t in range(total_steps):
            #We start off randomly exploring the environment until
            #we pass the number of initial exploration steps. 
            if t < self.start_exploration_steps:
                a = self.env.action_space.sample()
            else:
                if self.use_HER:
                    cato = np.concatenate((o,dg),0)
                    a = self.get_action(cato)
                else:
                    a = self.get_action(o)

            #Perform action in environment.
            obsNext, r, terminated, truncated, info = self.env.step(a)
            
            #Mujoco gymnasium environments return a dictionary, others arrays.
            if isinstance(obsNext, dict):
                o_next = obsNext['observation']
                #Goals if HER is enabled.
                if self.use_HER:
                    dg, ag = obsNext['desired_goal'], obsNext['achieved_goal']
            else:
                o_next = obsNext

            #Update episode return and length.                
            ep_ret += r
            ep_len += 1

            #We only want to be done if we hit the goal.
            done = False if (ep_len==self.max_ep_len or truncated) else terminated

            #Send experience to replay buffer.
            if self.use_HER:
                self.replay_buffer.store(o, a, r, o_next, done, dg, ag)
            else:
                self.replay_buffer.store(o, a, r, o_next, done)

            #Assign next observation to current .
            o = o_next

            #End of episode handling
            if terminated or truncated or (ep_len == self.max_ep_len):
                #Consider Tensorboard here.
                #logger.store(EpRet=ep_ret, EpLen=ep_len)

                #Run HER goal strategy against the most recept episode.
                if self.use_HER:
                   self.replay_buffer.run_goal_update_strategy(ep_len)
                
                #The episode is over, reset the environment.
                obs, info, ep_ret, ep_len = *self.env.reset(), 0, 0
                o = obs['observation'] if isinstance(obs, dict) else o

            #Update if past initial step threshold and on an update_every_steps multiple
            if t >= self.update_after and t % self.update_every_steps == 0:
                for j in range(self.update_every_steps):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(data=batch)

            #Increment epoch
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    self.save(self.seed)

                # Test the performance of the deterministic version of the agent.
                if self.run_tests:
                    self.test_agent(self.seed)

                #Use Tensoroboard here and log epoch info
                