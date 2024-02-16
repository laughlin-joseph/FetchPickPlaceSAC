import gymnasium as gym

class ImageEnv(gym.Wrapper):
    """
    Adds `image_observation`, `image_achieved_goal` and `image_desired_goal` fields
    to the observation dictionary. Used for rendering for now, and later for image-
    based tasks.
    """
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.env = env
        self.width = width
        self.height = height
        self.goal_img = None
        self.observation_space.spaces.update(
            image_observation=gym.spaces.Box(0, 255, (3, self.width, self.height)),
            image_desired_goal=gym.spaces.Box(0, 255, (3, self.width, self.height)),
            image_achieved_goal=gym.spaces.Box(0, 255, (3, self.width, self.height)),
        )
        self.reset()
        
    def reset(self):
        obs_dict = self.env.reset()
        initial_state = self.env.sim.get_state()
        goal = obs_dict['desired_goal']
        self._set_to_goal(goal);
        self.goal_img = self.env.render(mode='rgb_array', width=self.width, height=self.height).T.copy()
        self.env.sim.set_state(initial_state)
        
        obs_dict = self._update_obs_dict(obs_dict)
        
        return obs_dict
    
    def _update_obs_dict(self, obs_dict):
        obs_dict['image_desired_goal'] = self.goal_img
        obs_img = self.env.render(mode='rgb_array', width=self.width, height=self.height).T.copy()
        obs_dict['image_observation'] = obs_img
        obs_dict['image_achieved_goal'] = obs_img
        
        return obs_dict
        
    def _get_obs(self):
        obs_dict = self.env.env._get_obs()
        obs_dict = self._update_obs_dict(obs_dict)
        return obs_dict
    
    def step(self, act):
        obs_dict, rew, done, info = self.env.step(act)
        obs_dict = self._update_obs_dict(obs_dict)
        return obs_dict, rew, done, info
    
    def _set_to_goal(self, goal):
        """
        Goals are always xyz coordinates, either of gripper end effector or of object
        """
        if self.env.has_object:
            object_qpos = self.env.sim.data.get_joint_qpos('object0:joint')
            object_qpos[:3] = goal
            self.env.sim.data.set_joint_qpos('object0:joint', object_qpos)
        self.env.sim.data.set_mocap_pos('robot0:mocap', goal)
        self.env.sim.forward()
        for _ in range(100):
            self.env.sim.step()


class DoneOnSuccessWrapper(gym.Wrapper):
    def __init__(self, env):
        super(DoneOnSuccessWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = done or info.get('is_success', False)
        return obs, reward, done, info