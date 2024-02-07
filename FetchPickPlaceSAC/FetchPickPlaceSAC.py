import gymnasium as gym
from RLLib.Agents.SAC.Agent import SACAgent

#Create a fetch pick and place environment with dense rewards and pass it to a SAC Agent
env = gym.make("FetchPickAndPlaceDense-v2")

#Configure and train SACAgent
Agent = SACAgent(env)
Agent.train()

observation, info = env.reset()

#TODO: Gotta save the agent after training, create a new environment, and test.
#Now test the agent
for _ in range(1000):
    action = Agent.get_action(observation['observation'])
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()