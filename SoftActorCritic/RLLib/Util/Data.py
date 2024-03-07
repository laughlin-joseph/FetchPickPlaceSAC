from enum import Enum
import numpy as np
import operator
import random
import torch
import RLLib.Util.Functions as funcs
import RLLib.Util.Schedules as schedules

class SegmentTree:
    def __init__(self, size, operation, ne):
        assert size > 0 and size & (size - 1) == 0, "size must be positive and a power of 2."
        self._size = size
        self._value = [ne for _ in range(2 * size)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._size
        if end < 0:
            end += self._size
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._size - 1)

    def __setitem__(self, idx, val):
        idx += self._size
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._size
        return self._value[self._size + idx]

class SumSegmentTree(SegmentTree):
    def __init__(self, size):
        super().__init__(size=size, operation=operator.add, ne=0.0)

    def sum(self, start=0, end=None):
        return self.reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._size:
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._size

class MinSegmentTree(SegmentTree):
    def __init__(self, size):
        super().__init__(size=size, operation=min, ne=float('inf'))

    def min(self, start=0, end=None):
        return self.reduce(start, end)

class GoalUpdateStrategy(Enum):
    FINAL = 1
    FUTURE = 2
    EPISODE = 3

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(funcs.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(funcs.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = funcs.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = funcs.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trickS
        adv_mean  = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
            
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(funcs.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(funcs.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.o_next_buf = np.zeros(funcs.combined_shape(size, obs_dim), dtype=np.float32)
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

#HER buffer, see https://arxiv.org/pdf/1707.01495.pdf
#k is number of virtual copies per replayed step.
class HindsightExperienceReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, act_dim, goal_dim, size, device,
                 strat=GoalUpdateStrategy.FINAL, HER_obs_pr=lambda obs: None, HER_rew_func=lambda exp:0, k=4):
        super().__init__(obs_dim, act_dim, size, device)
        self.desired_goal_buf = np.zeros(funcs.combined_shape(size, goal_dim), dtype=np.float32)
        self.achieved_goal_buf = np.zeros(funcs.combined_shape(size, goal_dim), dtype=np.float32)
        self.strat = strat
        self.k = k
        self.HER_obs_pr = HER_obs_pr
        self.HER_rew_func = HER_rew_func
    
    def store(self, obs, act, rew, obs_next, done, desired_goal, achieved_goal):
        ReplayBuffer.store(self, obs, act, rew, obs_next, done)
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

        batch_last = batch_size - 1
        sample_end = batch_last - self.k
        final = process_list[batch_last]['ach']
        for pos, exp in enumerate(process_list):
            match self.strat:
                case GoalUpdateStrategy.FINAL:
                    self.store(exp['obs'],
                                exp['act'],
                                self.HER_rew_func(exp),
                                exp['o_next'],
                                exp['done'],
                                final,
                                exp['ach'])
                    
                case GoalUpdateStrategy.FUTURE:
                    if pos < (sample_end):
                        future_goal = exp['ach']
                        virtIndexes = np.random.randint(pos+1, high=sample_end+1, size=self.k)
                        for idx in virtIndexes:
                            self.store(process_list[idx]['obs'],
                                        process_list[idx]['act'],
                                        self.HER_rew_func(exp),
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
                                    self.HER_rew_func(exp),
                                    process_list[idx]['o_next'],
                                    process_list[idx]['done'],
                                    ep_goal,
                                    process_list[idx]['ach'])

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, act_dim, size, device, alpha, beta, total_steps):
        super().__init__(obs_dim, act_dim, size, device)
        assert alpha >= 0
        self.alpha = alpha
        
        self.beta_sched = schedules.LinearSchedule(total_steps, 1, beta)

        tree_size = 1
        while tree_size < size:
            tree_size *= 2

        self._tree_sum = SumSegmentTree(tree_size)
        self._tree_min = MinSegmentTree(tree_size)
        self._max_priority = 1.0
        
    def store(self, obs, act, rew, obs_next, done):
        tree_idx = (self.ptr+1) % self.max_size
        super().store(obs, act, rew, obs_next, done)
        self._tree_sum[tree_idx] = self._max_priority ** self.alpha
        self._tree_min[tree_idx] = self._max_priority ** self.alpha
    
    def sample_batch(self, step, beta: float = None, batch_size=50):        
        if beta is None:
            beta = self.beta_sched.get_step_val(step)
        assert (0 < beta and beta <= 1)
        indexes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._tree_min.min() / self._tree_sum.sum()
        max_weight = (p_min * self.size) ** (-beta)

        for index in indexes:
            p_sample = self._tree_sum[index] / self._tree_sum.sum()
            weight = (p_sample * self.size) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        indexes = np.array(indexes)
        
        batch = dict(obs=self.obs_buf[indexes],
                act=self.act_buf[indexes],
                rew=self.rew_buf[indexes],
                o_next=self.o_next_buf[indexes],
                done=self.done_buf[indexes],
                weights=weights, 
                indexes=indexes)
        
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in batch.items()}

    def update_priorities(self, indexes, priorities):
        assert len(indexes) == len(priorities)
        for index, priority in zip(indexes, priorities):
            assert priority > 0
            assert 0 <= index < self.size
            self._tree_sum[index] = priority ** self.alpha
            self._tree_min[index] = priority ** self.alpha

            self._max_priority = max(self._max_priority, priority)
            
    def _sample_proportional(self, batch_size):
        indexes = []
        p_total = self._tree_sum.sum(0, self.size - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            index = self._tree_sum.find_prefixsum_idx(mass)
            indexes.append(index)
        return indexes