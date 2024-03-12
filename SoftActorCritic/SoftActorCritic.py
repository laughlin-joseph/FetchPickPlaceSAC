import gymnasium as gym
from RLLib.Agents.SAC.Agent import SACAgent
from RLLib.Agents.PPO.Agent import PPOAgent

train_agent = True

ppo_env_str='CartPole-v1'
ppo_hidden=[512,512]
ppo_epoch_count=200
ppo_epoch_steps=12500
ppo_ep_len=250
ppo_w_HER=False
ppo_HER_obs_proc=lambda obs: None
ppo_HER_rew_function=lambda exp :0
ppo_w_PER=True
ppo_start_steps=12500
ppo_test_agent=True
ppo_log_agent=True
ppo_test_every=10

#sac_env_str='FetchPickAndPlace-v2'
#sac_hidden=[512,512]
#sac_epoch_count=200
#sac_epoch_steps=5000
#sac_ep_len=50
#sac_w_HER=True
#sac_HER_obs_proc=lambda obs: None
#sac_HER_rew_function=lambda exp :0
#sac_w_PER=False
#sac_start_steps=25000
#sac_test_agent=True
#sac_log_agent=True
#sac_test_every=10

sac_env_str='HandManipulateBlockRotateXYZDense-v1'
sac_hidden=[512,512,512]
sac_epoch_count=600
sac_epoch_steps=5000
sac_ep_len=50
sac_w_HER=False
sac_HER_obs_proc=lambda obs: None
sac_HER_rew_function=lambda exp :0
sac_w_PER=True
sac_start_steps=10000
sac_test_agent=True
sac_log_agent=True
sac_test_every=10

ppo_env=gym.make(ppo_env_str, max_episode_steps=ppo_ep_len)
sac_env=gym.make(sac_env_str, max_episode_steps=sac_ep_len)

if train_agent:
    #PPO_Agent=PPOAgent(
    #                ppo_env, 
    #                hidden_sizes=ppo_hidden,
    #                epochs=ppo_epoch_count,
    #                steps_per_epoch=ppo_epoch_steps,
    #                max_ep_len=ppo_ep_len,
    #                run_tests_and_record=ppo_test_agent, 
    #                enable_logging=ppo_log_agent,
    #                test_every_epochs=ppo_test_every
    #                )
    
    SAC_Agent=SACAgent(
                    sac_env, 
                    hidden_sizes=sac_hidden,
                    epochs=sac_epoch_count,
                    steps_per_epoch=sac_epoch_steps,
                    max_ep_len=sac_ep_len,
                    use_HER=sac_w_HER,
                    use_PER=sac_w_PER,
                    HER_obs_pr=sac_HER_obs_proc,
                    HER_rew_func=sac_HER_rew_function,
                    start_exploration_steps=sac_start_steps,
                    run_tests_and_record=sac_test_agent, 
                    enable_logging=sac_log_agent,
                    test_every_epochs=sac_test_every)
    
    #PPO_Agent.train()
    SAC_Agent.train()

#ppo_env.close()
sac_env.close()