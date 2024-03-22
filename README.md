# A repository for reinforcement learning exploration and development.  
  
## Purpose  
The development of this repository is intended to explore modern reinforcement learning algorithms  
using gymnasium environments, including gymnasium robotics environments:  
  
[Gymnasium: https://gymnasium.farama.org/index.html](https://gymnasium.farama.org/index.html)  
[Robotics: https://robotics.farama.org/index.html](https://robotics.farama.org/index.html)  
  
> - Developed using Visual Studio for managing projects and debugging code.  
> - Tensorboard is used for tracking model metrics during training,  
> graph visualization, and presenting environment renderings.  
> - Pytorch is used for building neural network modules.  
> - The control GUI is written using Kivy.  
> - Environments are managed using Anaconda.  
  
Aditional algorithms will be added over time and new research will be integrated as  
it becomes available and necessary.  
  
## Useage  
  
Development with Visual Studio is encouraged as that is what has been used so far.  
> **If running on a Windows OS use Python version 3.10.13 in a fresh Conda environment!**  
> Tensorflow stopped being supported in Windows some time ago  
> Using Anaconda to create an environment running Pythong version 3.10.13 will ensure  
> that other dependencies will install properly and run without conflict.  
  
### The following packages should be installed:  
 - Pytorch  
 - Tensorflow (For using Tensorboard.)  
 - Gymnasium  
 - Gymnasium-robotics  
 - Dill  (Operates on top of Pickle for more complex types.)  
 - OpenCV (For processing video files.)  
 - Pygame (Required for some Gymnasium environments.)  
 - Kivy  
  
## Execution  
  
The Core project has the file SimpleControl.py, this file serves as a simple control script for training  
agents. See the examples in the current file, but in general, one should instantiate their agent of choice,  
and then call the desired method on it.  
  
At the very least, take look at the agent constructors under CherryRL.Agents.<AgentOfChoice>.Agent.  
The code is meant to be read and the agents have quite a few configurable parameters.

Tensorboard can be automatically started during training to track model progress. Learning Curves, networkout  
output, and epoch returns are tracked, if testing is enabled rendered output of the test episode can also be  
captured and hosted on Tensorboard.  
  
Use the basic Tensorboard launcher app to start Tensorboard on saved data. Target the root of the logging directory  
that contains the desired data and hit the "Start Tensorboard" button. Tensorboard should start on the default port of 6006.  
I may add support to train, load, and save agents at a later date.  
  
## Generated Content  
  
Pickle files, Tensorboard logging data, and test sequence video files are saved in the project  
directory by default using a date and time naming convention.  
  
## Implemented so far  
**All agents trained for this experiment were run on a system with an AMD Ryzen 9 3900XT 12-Core Processor running above 4.0 GHz, 32 GB of 2666MHz RAM, and a Nvidia RTX 2080Ti**  
  
### Soft Actor Critic  
**Discrete:**  
Though initially designed for continuous environments this repo contains an implementation capable of handling them.  
Below are the results of running a SAC discrete agent against the CartPole-v1 gymnasium environment. The agent is stable and learns with time however it takes quite a while to train it.  
This experiment was run for 200 epochs, with 12500 steps per epoch, and 250 steps max per episode.  
Epoch wall time was roughly 40 minutes, under minimal system load outside of training.  
  
| Discrete SAC Learning Curve | Discrete SAC Critic | Discrete SAC Reward | CartPole Result |
| ----------- | ----------- | ----------- | ----------- |
| ![DISC SAC Learning Curve](https://github.com/laughlin-joseph/ProjectAssets/blob/master/CARTPOLE_SAC_DISC/LearningCurve.PNG) | ![DISC SAC Critic](https://github.com/laughlin-joseph/ProjectAssets/blob/master/CARTPOLE_SAC_DISC/CriticSignal.PNG) | ![DISC SAC Reward](https://github.com/laughlin-joseph/ProjectAssets/blob/master/CARTPOLE_SAC_DISC/Reward.PNG) | ![DISC SAC BEST RESULT](https://github.com/laughlin-joseph/ProjectAssets/blob/master/CARTPOLE_SAC_DISC/RenderSmall.gif) |  
  
SAC is an off policy algorithm, when trained with stochastic gradient descent past experiences are selected randomly over many episodes. This is useful when the "next step taken" matters more than understanding 
how an entire series of events affects a final outcome. E.g. navigating an obstacle course with moving obstacles vs balancing a pole in one's hand. In a highly dynamic environment the next step, or small set of steps matter quite a bit. As the environment changes so does the means by which the goal is achieved. Whereas with more static and predictable environment where a given sequence of events ultimately determines if a goal is acheived then understanding how a given sequence of events results in achieving a goal is more important.  
  
Therefore, it is my contention that just because SAC, an off policy algorithm, can be adapted for discrete environments it is still important to consider how the environment operates and to aim a degree of intuition and consideration at how an agent would best learn in it.  
  
It is always wise to learn from our failures.  
  
**Continuius**  
Soft Actor Critic was designed to handle high dimensionality action and observation spaces. It is an off policy algorithm with learns 2 Q functions and a policy for action selection.  
In this repository we have an implementation of SAC which has aceess to both a Hindsight Expereince Replay buffer and a Prioritized Experience Replay buffer.  
Here we explore using SAC with HER in a **sparse reward environment**, I.e. an environment where success reward signals are rare.  
Below are training results from running the SAC agent against mujoco robotics FetchPickAndPlace-v2 environment, a sparse reward environment.  
  
| Learning Curve Without HER | Reward Without HER |
| ----------- | ----------- |
| ![Learning curve without HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_SPARSE_NO_HER/LearningCurve.PNG) | ![Reward without HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_SPARSE_NO_HER/Reward.PNG) |  
  
Without the benefit of hindsight experience replay our agent fails to learn much after 200 epochs
  
Compare the above to the following:  
| Learning Curve With HER | Reward With HER |
| ----------- | ----------- |
| ![Learning curve with HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_SPARSE_HER/LearningCurve.PNG) |  ![Reward with HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_SPARSE_HER/Reward.PNG) |
  
As we can see, when using HER and running for the same number of epochs the agent learns relatively quickly.  
This is a genuiely impressive increase in sample efficiency.
  
Rendering of Competing Results:  
| Without HER | With HER |
| ----------- | ----------- |
| ![Without HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_SPARSE_NO_HER/RenderSmall.gif) | ![With HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_SPARSE_HER/RenderSmall.gif) |  

The results of using HER in a sparse reward environnments are rather clear. Instead of essentially being told that everything the agent does is wrong, with HER, the agent is able to learn from its mistakes.  
Much like a boxer practicing with focus mitts, the robotic agent learns to move its arm and manipulator to a given location even if that location is not the desired end goal.
  
### Proximal Policy Optimization  
**Discrete:**  
  
PPO is an on policy algorithm, this particular implementation computes reward-to-go and action advantage over the course of an entire epoch, however per-episode and even per-reward updates are possible according to the algorithm as described in the paper ![Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) under section 5 algorithm 1. In this experiment a PPO agent was trained in the CartPole-v1 for 200 epochs with 12500 steps per epoch, a max of 250 steps per episode. Each epoch ran for approximately 15 minutes, a significant improvement over discrete SAC.  
See the graphs below for a glimpse of how things went.
  
| Discrete PPO Learning Curve | Discrete PPO Reward | CartPole Result |
| ----------- | ----------- | ----------- |
| ![DISC PPO Learning Curve](https://github.com/laughlin-joseph/ProjectAssets/blob/master/CARTPOLE_PPO_DISC/LearningCurve.PNG) | ![DISC PPO Reward](https://github.com/laughlin-joseph/ProjectAssets/blob/master/CARTPOLE_PPO_DISC/Reward.PNG) | ![DISC PPO BEST RESULT](https://github.com/laughlin-joseph/ProjectAssets/blob/master/CARTPOLE_PPO_DISC/RenderSmall.gif) |  
  
PPO performs quite well in this environment, especially when compared with SAC's performance in the same environment, with the same applicable hyperparameters.  PPO achieves a higher net reward, learns with more stability and has increasing reward troughs. This learning stability combined with this particular agent's much improved wall time values makes training a PPO Agent an attractive solution for some environments.  
So far we've seen that SAC accels at operating in high dimensional continuous observation and action spaces where near term decisions matter more than future decisions. By contrast, PPO seems to handle more simple problems where starting conditions and subsequent actions all matter in particular sequence for the sake of acheiving a given goal and does so with a higher degree of sample and compute efficiency. At least, the aforementioned is true of this repository's implementations.
  
**Continuius**  

| Continuous Sparse PPO Learning Curve | Continuous Spsrse PPO Histogram | Continuous Sparse PPO Reward | Fetch Pick Place Sparse Result |
| ----------- | ----------- | ----------- | ----------- |
| ![CONT_SPARSE_PPO Learning Curve](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_PPO_SPARSE/LearningCurve.PNG) | ![CONT_SPARSE PPO Histogram](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_PPO_SPARSE/Histo.PNG) | ![CONT_SPARSE PPO Reward](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_PPO_SPARSE/Reward.PNG) | ![CONT_SPARSE PPO BEST RESULT](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_PPO_SPARSE/RenderSmall.gif) |  


| Continuous Dense PPO Learning Curve | Continuous Dense PPO Histogram | Continuous Dense PPO Reward | Fetch Pick Place Dense Result |
| ----------- | ----------- | ----------- | ----------- |
| ![CONT_DENSE_PPO Learning Curve](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_PPO_SPARSE/LearningCurve.PNG) | ![CONT_DENSE PPO Histogram](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_PPO_SPARSE/Histo.PNG) | ![CONT_DENSE PPO Reward](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_PPO_SPARSE/Reward.PNG) | ![CONT_DENSE PPO BEST RESULT](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_PPO_SPARSE/RenderSmall.gif) |  
  
### Special thanks  
[Farama Foundation](https://farama.org/)  
[OpeAI](https://github.com/openai)  
[Alexander Van de Kleut](https://github.com/avandekleut)  
[Petros Christodoulou](https://github.com/p-christ)  
[Haibin Zhou et. al.](https://github.com/coldsummerday)  
