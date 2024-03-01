import copy
import numpy as np
import datetime
import torch
import gymnasium.spaces as spaces
import RLLib.Agents.SAC.Core as core
import RLLib.Util.Functions as util

class SACAgent:
    def __init__(self, env, hidden_sizes=[512,512], seed=1,
                 epochs=200, steps_per_epoch=5000, max_ep_len=50, save_freq_epoch=10,
                 gamma=0.95, polyak=0.9995, lr=5e-4, temp_init=1.0, ent_pen_scale=0.5, q_clip=0.5, log_max=2, log_min=-10,
                 batch_size=1024, replay_buffer_size=int(1e6),
                 use_HER=True, HER_obs_pr=lambda obs: None, HER_rew_func=lambda exp:0, HER_strat=core.GoalUpdateStrategy.FUTURE, HER_k=1,
                 start_exploration_steps=5000, update_after_steps=10000, update_every_steps=100,
                 run_tests_and_record=False, enable_logging=False, test_every_epochs=10, done_at_goal=False):
        
        if enable_logging:
            self.log = enable_logging
            self._tboard_started = False
            #Add the summary writer to self.
            util.create_summary_writer(self)
            #Collect and clean input args.
            params = copy.copy(locals())
            env_name = env.spec.id
            params.pop('self', None)
            params.pop('env', None)
            params.update({'env_name': env_name})
            self.writer.add_text('ENV:', env_name)
            self.writer.add_text('Agent Parameters:',str(params))

        #Check for CUDA.
        self.device = util.get_device()

        #Envs.
        self.env = env
        self.env_name = self.env.spec.id
        util.set_dirs(self)
        
        #Set HER usage and buffer props.
        self.use_HER = use_HER
        self.HER_obs_pr = HER_obs_pr
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
        self.q_clip = q_clip
        self.ent_pen_scale = ent_pen_scale
        self.log_max = log_max
        self.log_min = log_min

        #Configure obs, act, and goal dims if required for HER.
        #See https://robotics.farama.org/envs/fetch/ for information regarding expected dims and types.        
        self.obs_not_dict = not isinstance(self.env.observation_space, spaces.dict.Dict)
        self.obs_dim, self.act_dim = util.get_environment_shape(self)

        if self.use_HER:
            if self.obs_not_dict:
                if self.HER_obs_pr is None:
                    raise ValueError('Cannot use HER without goal information. HER_obs_pr returns None.')
                else:
                    test_ob = torch.rand(tuple(self.obs_dim)).uniform_(-1, 1)
                    res = self.HER_obs_pr(test_ob)
                    if not isinstance(res, tuple):
                        raise ValueError('HER_obs_pr must return a tuple containing 2 numpy arrays: desired goal, and achieved goal.')
                    self.goal_dim= np.array(res[0].shape)
            else:
                self.goal_dim = np.array(self.env.observation_space['desired_goal'].shape)
            
            self.net_obs_dim = self.goal_dim + self.obs_dim
        else:
            self.net_obs_dim = self.obs_dim

        #Epochs and episode length
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.max_ep_len = max_ep_len
        self.save_freq_epoch = save_freq_epoch

        #Initial exploration and update cycle management.
        self.start_exploration_steps = start_exploration_steps
        self.update_after_steps = update_after_steps
        self.update_every_steps = update_every_steps

        #Set all seeds.
        util.set_seed(seed)

        #Temp tuning setup
        self.log_temp = torch.tensor(np.log(self.temp_init)).to(self.device)
        self.log_temp.requires_grad = True
        #See https://arxiv.org/pdf/2209.10081.pdf Notes following equation 10 in section 3.
        if self.action_discrete:
            self.target_entropy = -np.log((1.0 / self.num_discrete_actions)) * 0.98
        else:
            self.target_entropy = -self.act_dim[0]

        #Create actor critic networks and freeze targets.
        #For discrete SAC see the following paper: https://arxiv.org/pdf/1910.07207.pdf
        self.ac = core.MLPActorCritic(self.net_obs_dim, self.act_dim, hidden_sizes,
                                      discrete=self.action_discrete, num_dis_actions=self.num_discrete_actions,
                                      log_max=self.log_max, log_min=self.log_min)
        self.ac.to(self.device)

        if self.log:
            obs_test = torch.rand(tuple(self.net_obs_dim)).uniform_(-1, 1).unsqueeze(0).to(self.device)
            act_test = torch.rand(tuple(self.act_dim)).uniform_(-1, 1).unsqueeze(0).to(self.device)
            self.piwriter.add_graph(self.ac.pi, obs_test)
            if self.action_discrete:
                self.qwriter.add_graph(self.ac.q1, [[obs_test]])
            else:
                self.qwriter.add_graph(self.ac.q1, [[obs_test, act_test]])

        #Optim for trained networks actor, critic1, and critic2. Optim for temp dual func.
        self.pi_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.ac.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.ac.q2.parameters(), lr=lr)
        self.log_temp_optimizer = torch.optim.Adam([self.log_temp], lr=lr)

        #Experience replay buffer.
        self.configure_buffer()

        #Test env wrap for recording and test data loading.
        self.run_tests_and_record = run_tests_and_record
        self.test_every_epochs = test_every_epochs
        self.done_at_goal = done_at_goal
        self.test_count = 0
        if self.run_tests_and_record:
            util.setup_test_env(self, 'TestRecordings')

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('env', None)
        state.pop('test_env', None)
        state.pop('replay_buffer', None)
        state.pop('writer', None)
        state.pop('piwriter', None)
        state.pop('qwriter', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def temp(self):
        return self.log_temp.exp()
   
    #Minimize the Bellman residual.
    #Section 4.2 Equation 5 of https://arxiv.org/pdf/1812.05905v2.pdf
    #Q(st, at) - (r(st, at) + gamma * V(st+1))
    #Where V(st+1)= (Q(st+1, at+1) - temp * log(pi(at+1)))
    def compute_loss_q(self, data):
        o, a, r, o_next, done = data['obs'], data['act'], data['rew'], data['o_next'], data['done']

        a_next, probs_next = self.ac.pi(o_next, deterministic=False)

        #Compute MBSE.
        if self.ac.pi.discrete:
            expected_q1 = self.ac.q1([o])
            expected_q2 = self.ac.q2([o])
            #No backprop through these components, targq params receive soft updates.
            with torch.no_grad():    
                #For info regarding double average Q nets with 
                #clipping see section 5.2 of the following: https://arxiv.org/pdf/2209.10081.pdf
                q1_pi_targ = self.ac.q1targ([o_next])
                q2_pi_targ = self.ac.q2targ([o_next])
                #We get exact entropy and probability of actions due to discrete action space.
                logp_a_next = probs_next[0]
                probs_a_next = probs_next[1]
                #Use average Q instead of min in order to avoid Q value undereestimation.
                q_pi_targ_avg = torch.mean(torch.stack([q1_pi_targ, q2_pi_targ], axis=-1), axis=-1)
                #Calculate value for target using average Q.
                target_val = (probs_a_next * (q_pi_targ_avg - self.temp.detach() * logp_a_next)).sum(axis=-1)
                backup = r + self.gamma * (1 - done) * target_val
            #Add clipped difference in order to avoid underavalue Q, take max later if targ > expected.
            clipped_q1 = (q1_pi_targ + torch.clamp(expected_q1 - q1_pi_targ, -self.q_clip, self.q_clip)).sum(axis=-1)
            clipped_q2 = (q2_pi_targ + torch.clamp(expected_q2 - q2_pi_targ, -self.q_clip, self.q_clip)).sum(axis=-1)
            #Sum expected Q.                
            expected_q1 = expected_q1.sum(axis=-1)
            expected_q2 = expected_q2.sum(axis=-1)
            #Take loss from larger Q1 val.
            loss_q1_ex = ((expected_q1 - backup)**2).mean()
            loss_q1_clip = ((clipped_q1 - backup)**2).mean()
            loss_q1 = torch.max(loss_q1_ex, loss_q1_clip)
            #Take loss from larger Q2 val.
            loss_q2_ex = ((expected_q2 - backup)**2).mean()
            loss_q2_clip = ((clipped_q2 - backup)**2).mean()                
            loss_q2 = torch.max(loss_q2_ex, loss_q2_clip)
        else:
            expected_q1 = self.ac.q1([o,a])
            expected_q2 = self.ac.q2([o,a])
            #No backprop through these components, targq params receive soft updates.
            with torch.no_grad():
                #Get Target TD action from next observation.
                q1_pi_targ = self.ac.q1targ([o_next, a_next])
                q2_pi_targ = self.ac.q2targ([o_next, a_next])
                q_pi_targ_min = torch.min(q1_pi_targ, q2_pi_targ)
                logpi_a_next = probs_next[0]
                target_val = (q_pi_targ_min - self.temp * logpi_a_next)
                backup = r + self.gamma * (1 - done) * target_val
            loss_q1 = ((expected_q1 - backup)**2).mean()
            loss_q2 = ((expected_q2 - backup)**2).mean()

        #Sum average loss and return.
        loss_q = loss_q1 + loss_q2
        return loss_q

    #Function to return pi loss.
    def compute_loss_pi(self, data):
        old_ent = self.ac.pi.entropy
        o = data['obs']
        pi_act, probs = self.ac.pi(o, deterministic=False)
        current_ent = self.ac.pi.entropy
        if self.ac.pi.discrete:
            q1_pi = self.ac.q1([o])
            q2_pi = self.ac.q2([o])
            logprob_pi = probs[0]
            pi_probs = probs[1]
            #For information on discrete SAC with entropy
            #penalty see section 5.1 equation 14: https://arxiv.org/pdf/2209.10081.pdf
            q_pi_avg = torch.mean(torch.stack([q1_pi, q2_pi], axis=-1), axis=-1)
            ent_pen = ((current_ent - old_ent)**2).mean()
            scaled_ent_pen = self.ent_pen_scale * ent_pen
            #Calculate actor loss and add entropy penalty.
            loss_pi = (pi_probs * (self.temp.detach() * logprob_pi - q_pi_avg)).sum(axis=-1).mean()
            loss_pi += scaled_ent_pen
        else:
            #See Section 4.2 Equations 7, 8, and 9 of https://arxiv.org/pdf/1812.05905v2.pdf
            # Max V(st) = (Q(st, at) - temp * log(pi(at)))
            # by min (temp * log((pi(atR))) - Q(st, atR)
            # where atR is pi's reparametrized  output.
            # Pi outputs a mu and std, actions are selected  by
            # sampling a value from a standard normal distribution.
            # This value is then scaled and shifted by pi's output, the reparameterization trick.
            q1_pi = self.ac.q1([o, pi_act])
            q2_pi = self.ac.q2([o, pi_act])
            q_pi = torch.min(q1_pi, q2_pi)
            logprob_pi = probs[0]
            loss_pi = (self.temp.detach() * logprob_pi - q_pi).mean()
        
        return loss_pi, probs
    
    def configure_buffer(self):
        if self.use_HER:
            self.replay_buffer = core.HERReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, goal_dim=self.goal_dim, size=self.replay_buffer_size,device=self.device,
                                                      strat=self.HER_strat, HER_obs_pr=self.HER_obs_pr, HER_rew_func=self.HER_rew_func, k=self.HER_k)
        else:
            self.replay_buffer = core.SACReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_buffer_size, device=self.device)

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
        loss_pi, probs = self.compute_loss_pi(data)
        #Get partials.
        loss_pi.backward()
        #Take GD step for actor net pi.
        self.pi_optimizer.step()

        #Adjust the temp parameter. Detach logprob_pi and target_entropy from optim.
        #See section 5 equation 17 of https://arxiv.org/pdf/1812.05905v2.pdf
        if self.action_discrete:
            logprob_pi = probs[0]
            pi_probs = probs[1]
            inner =  (-logprob_pi - self.target_entropy).detach()
            temp_loss = (pi_probs.detach() * (self.temp * inner)).sum(axis=-1).mean()
        else:
            logprob_pi = probs[0]
            inner = (-logprob_pi - self.target_entropy).detach()
            temp_loss = (self.temp * inner).mean()
        
        self.log_temp_optimizer.zero_grad()
        temp_loss.backward()
        self.log_temp_optimizer.step()

        #Unfreeze the Q Networks.
        for q in (self.ac.q1, self.ac.q2):
            util.freeze_thaw_parameters(q, freeze=False)
        
        #For the soft part of SAC, Polyak average the Target Q networks.        
        with torch.no_grad():
            for q, qt in [(self.ac.q1, self.ac.q1targ),(self.ac.q2, self.ac.q2targ)]:
                for p, p_targ in zip(q.parameters(), qt.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

        return loss_q.detach(), loss_pi.detach(), temp_loss.detach()
                
    def get_action(self, o, deterministic=False):
        shaped = torch.as_tensor(o.reshape(1, *self.net_obs_dim), dtype=torch.float32, device=self.device)
        action = self.ac.act(shaped, deterministic)
        if self.action_discrete:
            action = int(action)
        else:
            action = action.reshape(*self.act_dim)
        return action

    def test_agent(self):
        #Increment agent test_count
        reason = ''
        ep_rew = 0
        self.test_count += 1
        #Reset test environment.
        obs, info = self.test_env.reset()
        a, r, terminated, truncated, dg = [], 0, False, False, []
        o = obs if self.obs_not_dict else obs['observation']

        #Begin testing.
        for _ in range(self.max_ep_len):
            if self.use_HER:
                if self.obs_not_dict:
                    res = self.replay_buffer.HER_obs_pr(obs)
                    dg = res[0]
                else:
                    dg = obs['desired_goal']
                cato = np.concatenate((o,dg),0)
                a = self.get_action(cato)
            else:
                a = self.get_action(o)
            
            obs, r, terminated, truncated, info = self.test_env.step(a)             
            o = obs if self.obs_not_dict else obs['observation']
            ep_rew += r

            if self.done_at_goal and info.get('is_success', False):
                reason = 'Done'
                break
            if terminated or truncated:
                reason = 'Truncated' if truncated else 'Terminated'
                break
        if reason:
            print('\n%s condition reached during testing.' % reason)
        
        return ep_rew, info
            
    def train(self):
        total_steps = self.steps_per_epoch * self.epochs
        start_time, epoch = datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), 0
        obs_Next, info = self.env.reset()
        ep_len, ep_q_loss, ep_pi_loss, ep_temp_loss = 0, 0, 0, 0
        a, r, o_next, done = [], 0, [], 0
        test_rew, test_info = 0, None
        terminated, truncated, dg, ag = False, False, [], []
        o = obs_Next if self.obs_not_dict else obs_Next['observation']
        for t in range(total_steps):

            #We start off randomly exploring the environment until
            #we pass the number of initial exploration steps. 
            if t < self.start_exploration_steps:
                a = self.env.action_space.sample()
            else:
                if self.use_HER:
                    #    |\__/,|   (`\
                    #  _.|o o  |_   ) )
                    #-(((---(((--------
                    cato = np.concatenate((o,dg),0)
                    a = self.get_action(cato)
                else:
                    a = self.get_action(o)

            #Perform action in environment.
            obs_Next, r, terminated, truncated, info = self.env.step(a)
            
            #Handle Mujoco environment obs.
            o_next = obs_Next if self.obs_not_dict else obs_Next['observation']
            if self.use_HER:
                if self.obs_not_dict:
                    res  = self.replay_buffer.HER_obs_pr(obs_Next)
                    dg, ag = res[0], res[1]
                else:
                    dg, ag = obs_Next['desired_goal'], obs_Next['achieved_goal']

            #Update episode return and length.                
            ep_len += 1

            #We only want to be done if we hit the goal.
            if self.done_at_goal and info.get('is_success', False):
                done = 1
                
            #Send experience to replay buffer.
            if self.use_HER:
                self.replay_buffer.store(o, a, r, o_next, done, dg, ag)
            else:
                self.replay_buffer.store(o, a, r, o_next, done)

            #Assign next observation to current .
            o = o_next

            #End of episode handling
            if terminated or truncated or done or (ep_len == self.max_ep_len):
                #Run HER goal strategy against the most recept episode.
                if self.use_HER:
                   self.replay_buffer.run_goal_update_strategy(ep_len)
                
                #The episode is over, reset the environment.
                obs_Next, info = self.env.reset()
                r, terminated, truncated, ep_len = 0, False, False, 0
                o = obs_Next if self.obs_not_dict else obs_Next['observation']

            #Update if past initial step threshold and on an update_every_steps multiple
            if t >= self.update_after_steps and t % self.update_every_steps == 0:
                for j in range(self.update_every_steps):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    ep_q_loss, ep_pi_loss, ep_temp_loss = self.update(data=batch)

            if (t+1) % self.steps_per_epoch == 0:
                #Save model
                if (epoch % self.save_freq_epoch == 0) or (epoch == self.epochs):
                    util.save(self, self.env_name)
                
                #Test the performance of the deterministic actor.
                if (epoch % self.test_every_epochs == 0) and  self.run_tests_and_record:
                    test_rew, test_info = self.test_agent()
                    video_dir = self.test_env.spec.additional_wrappers[0].kwargs['video_folder']
                    frames, rate = util.get_latest_frames(video_dir, 'mp4')
                    self.writer.add_video('Recording of latest test:', frames, global_step=epoch, fps=rate)


                if self.log:
                    #Loss Scalars
                    self.writer.add_scalar('Total return', test_rew, global_step=epoch)
                    self.writer.add_scalars('Actor/Critic loss', {'Actor Loss':ep_pi_loss, 'Critic Loss':ep_q_loss}, global_step=epoch)
                    self.writer.add_scalar('Total Temp loss', ep_temp_loss, global_step=epoch)
                    #Learned Values
                    self.writer.add_scalar('Temp/Alpha', self.temp.detach().item(), global_step=epoch)
                    #Histograms
                    if self.action_discrete:
                        test_o = torch.as_tensor(o.reshape(1, *self.net_obs_dim), dtype=torch.float32, device=self.device)
                        _, probs = self.ac.pi(test_o, deterministic=False)
                        histo_vals = util.sample_categorical(probs[1])
                    else:
                        self.writer.add_scalar('Mu Average', self.ac.pi.mu.mean(axis=1), global_step=epoch)
                        self.writer.add_scalar('Sigma/std_dev Average', self.ac.pi.std.mean(axis=1), global_step=epoch)
                        histo_vals = util.sample_normal(self.ac.pi.mu, self.ac.pi.std)
                    
                    self.writer.add_histogram('Action Sampling Distribution', histo_vals, global_step=epoch)
                    if not self._tboard_started:
                        running, board_url = util.start_tensorboard(self.log_data_dir)
                        self._tboard_started = running
                        self.tensor_board_url = board_url
                    #Write all to Tensorboard.
                    self.writer.flush()
                #Increment epoch
                epoch = (t+1) // self.steps_per_epoch
        if self.log:
            finish_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            print('\nTraining complete!\nStarted:%s\nFinished:%s' % (start_time, finish_time))
            self.writer.close()
                