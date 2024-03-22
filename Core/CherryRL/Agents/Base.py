from abc import ABC
import torch
import CherryRL.Util.Functions as funcs

class BaseAgent(ABC):
    def __init__(self, env, seed=1,
                 epochs=200, steps_per_epoch=5000, max_ep_len=50, save_freq_epoch=10,
                 add_goal_to_obs=False, use_HER=False, use_PER=False,
                 run_tests_and_record=False, test_every_epochs=10, done_at_goal=False):
        super(BaseAgent, self).__init__()
        
        #Set support for HER and PER
        self.use_HER = use_HER
        self.use_PER = use_PER

        #Check for CUDA.
        self.device = funcs.get_device()
        
        #Check if agent should be goal aware.
        self.add_goal_to_obs = add_goal_to_obs

        #Envs.
        self.env = env
        self.env_name = self.env.spec.id
        funcs.set_dirs(self)

        #Get and set environment dim information.
        funcs.get_environment_shape(self)

        #Set all seeds.
        funcs.set_seed(seed)
        
        #Epochs and episode length
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.max_ep_len = max_ep_len
        self.save_freq_epoch = save_freq_epoch
        
        #Test env wrap for recording and test data loading.
        self.run_tests_and_record = run_tests_and_record
        self.test_every_epochs = test_every_epochs
        self.done_at_goal = done_at_goal
        self.test_count = 0
        if self.run_tests_and_record:
            funcs.setup_test_env(self, 'TestRecordings')
            
    def test_agent(self):
        #Increment agent test_count
        reason = ''
        ep_rew = 0
        self.test_count += 1
        #Reset test environment.
        obs, info = self.test_env.reset()
        a, r, terminated, truncated = [], 0, False, False
        o = funcs.process_observation(self, obs)
        #Begin testing.
        for _ in range(self.max_ep_len):
            a = self.ac.act(o)
            obs, r, terminated, truncated, info = self.test_env.step(a)             
            o = funcs.process_observation(self, obs)
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