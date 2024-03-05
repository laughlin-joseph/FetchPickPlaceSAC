from enum import Enum
import numpy as np
import operator
import random
import torch
import RLLib.Util.Functions as util

class SegmentTree():
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
        super().__init__(size=size, operation=operator.add, neutral_element=0.0)

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
        super().__init__(size=size, operation=min, neutral_element=float('inf'))

    def min(self, start=0, end=None):
        return self.reduce(start, end)

class GoalUpdateStrategy(Enum):
    FINAL = 1
    FUTURE = 2
    EPISODE = 3
            
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(util.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(util.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.o_next_buf = np.zeros(util.combined_shape(size, obs_dim), dtype=np.float32)
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
        self.desired_goal_buf = np.zeros(util.combined_shape(size, goal_dim), dtype=np.float32)
        self.achieved_goal_buf = np.zeros(util.combined_shape(size, goal_dim), dtype=np.float32)
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
    def __init__(self, obs_dim, act_dim, size, device, alpha):
        super().__init__(obs_dim, act_dim, size, device)
        assert alpha >= 0
        self.alpha = alpha
    
        tree_size = 1
        while tree_size < size:
            tree_size *= 2

        self._tree_sum = SumSegmentTree(tree_size)
        self._tree_min = MinSegmentTree(tree_size)
        self._max_priority = 1.0
        
    def store(self, obs, act, rew, obs_next, done):
        tree_idx = (self.ptr+1) % self.max_size
        super().store(self, obs, act, rew, obs_next, done)
        self._tree_sum[tree_idx] = self._max_priority ** self._alpha
        self._tree_min[tree_idx] = self._max_priority ** self._alpha
    
    def sample_batch(self, beta=.5, batch_size=50):
        assert beta > 0
        indexes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._tree_min.min() / self._tree_sum.sum()
        max_weight = (p_min * self.size) ** (-beta)

        for index in indexes:
            p_sample = self._tree_sum[index] / self._tree_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        
        batch = dict(obs=self.obs_buf[indexes],
                act=self.act_buf[indexes],
                rew=self.rew_buf[indexes],
                o_next=self.o_next_buf[indexes],
                done=self.done_buf[indexes],
                weights=weights, 
                indexes=np.array(indexes))
        
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in batch.items()}

    def update_priorities(self, indexes, priorities):
        assert len(indexes) == len(priorities)
        for index, priority in zip(indexes, priorities):
            assert priority > 0
            assert 0 <= index < self.size
            self._tree_sum[index] = priority ** self._alpha
            self._tree_min[index] = priority ** self._alpha

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