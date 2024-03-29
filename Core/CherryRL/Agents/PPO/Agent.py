import copy
import datetime
import gymnasium
import gymnasium.spaces as spaces
import time
import torch
from torch import nn
from torch.optim import Adam
import CherryRL.Agents.Base as Base
import CherryRL.Agents.PPO.Nets as nets
import CherryRL.Util.Functions as funcs
import CherryRL.Util.Data as data

class PPOAgent(Base.BaseAgent):
    def __init__(self, env, hidden_sizes=[512,512], seed=1,
        epochs=200, steps_per_epoch=5000, max_ep_len=50, save_freq_epoch=10,
        gamma=0.99, clip_ratio=0.2, pi_lr=3e-4, valfunc_lr=1e-3,
        add_goal_to_obs=False, train_pi_iters=80, train_valfunc_iters=80, lam=0.97, target_kl=0.01,
        run_tests_and_record=False, enable_logging=False, test_every_epochs=10, done_at_goal=False):
        
        super(PPOAgent, self).__init__(env, seed,
                                       epochs, steps_per_epoch, max_ep_len, save_freq_epoch,
                                       add_goal_to_obs, False, False,
                                       run_tests_and_record, test_every_epochs, done_at_goal)

        #Setup Tensorboard.
        if enable_logging:
            self.log = enable_logging
            self._tboard_started = False
            #Add the summary writer to self.
            funcs.create_summary_writer(self)
            #Collect and clean input args.
            params = copy.copy(locals())
            env_name = env.spec.id
            params.pop('self', None)
            params.pop('env', None)
            params.update({'env_name': env_name})
            self.writer.add_text('ENV:', env_name)
            self.writer.add_text('Agent Parameters:',str(params))

        #Set params and constants.
        self.gamma = gamma
        self.lam = lam
        self.pi_lr = pi_lr
        self.valfunc_lr = valfunc_lr
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_valfunc_iters = train_valfunc_iters
        self.target_kl = target_kl

        #Create AC nets.
        self.ac = nets.MLPActorCritic(self, hidden_sizes, activation=nn.Tanh)    
        self.ac.to(self.device)            

        #Create PPO buffer, set size to steps_per_epoch for online training.
        self.epoch_buffer = data.PPOBuffer(self.net_obs_dim, self.act_dim, steps_per_epoch, self.device, gamma, lam)

        #Optim for trained nets.
        self.valfunc_optimizer = Adam(self.ac.value.parameters(), lr=valfunc_lr)
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('env', None)
        state.pop('test_env', None)
        state.pop('epoch_buffer', None)
        state.pop('writer', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    #Policy loss function.
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        #Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        #Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    #Value loss function.
    def compute_loss_value(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.value(obs) - ret)**2).mean()

    def update(self):
        data = self.epoch_buffer.get()
        loss_pi, loss_value = 0,0
        
        #Train Policy.
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                #Consider increasing target_kl if this is hit often and learning curves are flat.
                print('INFO: Early termination at step %d due to reaching max divergence.'%i)
                break
            loss_pi.backward()
            self.pi_optimizer.step()

        #Value function learning.
        for i in range(self.train_valfunc_iters):
            self.valfunc_optimizer.zero_grad()
            loss_value = self.compute_loss_value(data)
            loss_value.backward()
            self.valfunc_optimizer.step()
        
        return loss_pi, loss_value
    
    def train(self):
        # Prepare for interaction with environment
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        obs, info = self.env.reset()
        o  = funcs.process_observation(self, obs)
        ep_ret, ep_len = 0,0

        #Run each epoch
        for epoch in range(self.epochs):
            for t in range(self.steps_per_epoch):
                a, val, logp = self.ac.step(o)

                obs_next, r, terminated, truncated, info = self.env.step(a)
                o_next  = funcs.process_observation(self, obs_next)
                ep_ret += r
                ep_len += 1

                #Send experience to replay buffer.
                self.epoch_buffer.store(o, a, r, val, logp)
            
                #Assign next observation to current.
                o = o_next

                epoch_ended = t==self.steps_per_epoch-1

                if terminated or truncated or epoch_ended:
                    if epoch_ended and not(terminated or truncated):
                        print('INFO: Premature episode end due to end of epoch at %d steps.' % ep_len)
                        _, val, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))                        
                    else:
                        val = 0
                    self.epoch_buffer.finish_path(val)
                    obs, info = self.env.reset()
                    o  = funcs.process_observation(self, obs)
                    ep_ret, ep_len = 0,0

            #Update
            ep_loss_pi, ep_loss_value = self.update()       
            
            #Save model
            if (epoch % self.save_freq_epoch == 0) or (epoch == self.epochs):
                funcs.save(self, self.env_name)
                
            #Test the performance of the deterministic actor.
            if (epoch % self.test_every_epochs == 0) and  self.run_tests_and_record:
                test_rew, test_info = self.test_agent()
                video_dir = self.test_env.spec.additional_wrappers[0].kwargs['video_folder']
                frames, rate = funcs.get_latest_frames(video_dir, 'mp4')
                self.writer.add_video('Recording of latest test:', frames, global_step=epoch, fps=rate)

            if self.log:
                #Loss Scalars
                self.writer.add_scalar('Total return', test_rew, global_step=epoch)
                self.writer.add_scalars('Actor/Critic loss', {'Actor Loss':ep_loss_pi, 'Critic Loss':ep_loss_value}, global_step=epoch)
                #Histograms
                if self.action_discrete:
                    test_o = torch.as_tensor(o.reshape(1, *self.obs_dim), dtype=torch.float32, device=self.device)
                    dist, _ = self.ac.pi(test_o)
                    histo_vals = dist.sample((100,))
                else:
                    self.writer.add_scalar('Mu Average', self.ac.pi.mu.mean(), global_step=epoch)
                    self.writer.add_scalar('Sigma/std_dev Average', self.ac.pi.std.mean(), global_step=epoch)
                    histo_vals = funcs.sample_normal(self.ac.pi.mu, self.ac.pi.std)
                    
                self.writer.add_histogram('Action Sampling Distribution', histo_vals, global_step=epoch)
                if not self._tboard_started:
                    running, board_url = funcs.start_tensorboard(self.log_data_dir)
                    self._tboard_started = running
                    self.tensor_board_url = board_url
                #Write all to Tensorboard.
                self.writer.flush()
        if self.log:
            finish_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            print('\nTraining complete!\nStarted:%s\nFinished:%s' % (start_time, finish_time))
            self.writer.close()