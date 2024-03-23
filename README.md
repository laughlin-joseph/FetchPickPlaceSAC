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
  
Additional algorithms will be added over time and new research will be integrated as  
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
  
The Core project has the file SimpleControl.py, this file serves as a simple control script for training agents. See the examples in the current file, but in general, one should instantiate their agent of choice, and then call the desired method on it.  
  
At the very least, take a look at the agent constructors under CherryRL.Agents.**AgentOfChoice**.Agent. The code is meant to be read and the agents have quite a few configurable parameters.  
  
Tensorboard can be automatically started during training to track model progress. Learning Curves, action histograms, network output, and epoch returns are tracked, if testing is enabled rendered output of the latest test episode can also be captured and hosted on Tensorboard.  
  
Use the basic Tensorboard launcher app to start Tensorboard on saved data. Target the root of the logging directory that contains the desired data and hit the "Start Tensorboard" button. Tensorboard should start on the default port of 6006. I may add support to train, load, and save agents at a later date.  
  
## Generated Content  
  
Pickle files, Tensorboard logging data, and test sequence video files are saved in the project directory by default using a date and time naming convention.  
  
## Implementations and Experiments
> **All agents trained for this experiment were run on a system with an AMD Ryzen 9 3900XT 12-Core Processor running above 4.0 GHz, 32 GB of 2666MHz RAM, and a Nvidia RTX 2080Ti**  
  
### Soft Actor Critic 
  
**Discrete:**  
  
Though initially designed for continuous environments this repo contains an implementation capable of handling them. Below are the results of running a SAC discrete agent against the CartPole-v1 gymnasium environment. The agent is stable and learns with time however it takes quite a while to train it.  
  
> This experiment was run for 200 epochs, with 12500 steps per epoch, and 250 steps max per episode. Epoch wall time was roughly 40 minutes, under minimal system load outside of training.  
  
| Discrete SAC Learning Curve | Discrete SAC Critic | Discrete SAC Reward | CartPole Result |
| ----------- | ----------- | ----------- | ----------- |
| ![DISC SAC Learning Curve](https://github.com/laughlin-joseph/ProjectAssets/blob/master/CARTPOLE_SAC_DISC/LearningCurve.PNG) | ![DISC SAC Critic](https://github.com/laughlin-joseph/ProjectAssets/blob/master/CARTPOLE_SAC_DISC/CriticSignal.PNG) | ![DISC SAC Reward](https://github.com/laughlin-joseph/ProjectAssets/blob/master/CARTPOLE_SAC_DISC/Reward.PNG) | ![DISC SAC BEST RESULT](https://github.com/laughlin-joseph/ProjectAssets/blob/master/CARTPOLE_SAC_DISC/RenderSmall.gif) |  
  
SAC is an off-policy algorithm, when trained with stochastic gradient descent past experiences are selected randomly over many episodes. This is useful when the "next step taken" matters more than understanding how an entire series of events affects an outcome. E.g. navigating an obstacle course with moving obstacles vs balancing a pole in one's hand. In a highly dynamic environment the next step, or small set of steps matters quite a bit. As an environment changes so do the means by which a goal is achieved. Whereas with a static and more predictable environment where a given sequence of events ultimately determines if a goal is achieved then understanding how a given sequence of events results in achieving a goal is more important.  
  
Therefore, I contend that just because SAC, an off-policy algorithm, can be adapted for discrete environments it is still important to consider how the environment operates and to aim a degree of intuition and consideration at how an agent would best learn in it.  
  
It is always wise to learn from our failures.  
  
**Continuius**  
  
Soft Actor Critic was designed to handle high dimensionality action and observation spaces. It is an off-policy algorithm that learns 2 Q functions and a policy for action selection. SAC is an entropy regularized algorithm, it adds the scaled entropy of a selected action to the Q value for said action and attempts to maximize both the quality of the action and the entropy of the distribution it was selected from. This is done to maximize expected return and randomness while operating in and exploring an environment. A temperature parameter is used to scale randomness as the agent learns. 
  
In this repository, we have an implementation of SAC with automatic temp tuning where alpha is a learned parameter, access to both Hindsight Experience Replay and Prioritized Experience Replay buffers, and target entropy values for temp tuning. Here we explore using SAC with HER in a **sparse reward environment**, i.e. an environment where success reward signals are rare.  
  
Below are training results from running the SAC agent against mujoco robotics FetchPickAndPlace-v2 environment, a sparse reward environment by default with the option to configure a dense reward signal.  
  
| Learning Curve Without HER | Action Histogram Without HER | Reward Without HER |
| ----------- | ----------- | ----------- |
| ![Learning curve without HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_SPARSE_NO_HER/LearningCurve.PNG) | ![Histogram Without HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_SPARSE_NO_HER/Histo.PNG) |![Reward without HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_SPARSE_NO_HER/Reward.PNG) |  
  
Without the benefit of hindsight experience replay our agent fails to learn much after 200 epochs. Rewards are indeed rather sparse, and the distribution of selected actions remains wide over the course of training. While the agent does learn it isn't very sample efficient.  

Compare the above to the following:  
  
| Learning Curve With HER | Action Histogram With HER  | Reward With HER |
| ----------- | ----------- | ----------- |
| ![Learning curve with HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_SPARSE_HER/LearningCurve.PNG) | ![Histogram With HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_SPARSE_HER/Histo.PNG) | ![Reward with HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_SPARSE_HER/Reward.PNG) |
  
As we can see, when using HER and running for the same number of epochs the agent learns relatively quickly. This is a genuinely impressive increase in sample efficiency, reward, and overall performance. Rewards stabilize, convergence is nearly achieved, and the distribution of action selection suggests developing certainty.
  
Rendering of Competing Results:  
  
| Without HER | With HER |
| ----------- | ----------- | 
| ![Without HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_SPARSE_NO_HER/RenderSmall.gif) | ![With HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_SPARSE_HER/RenderSmall.gif) |  

The result of using HER in sparse reward environments is rather clear. Instead of essentially being told that nearly everything the agent does is wrong, with HER, the agent is able to learn from its mistakes.  
  
Much like a boxer practicing with focus mitts, the robotic agent learns to move its arm and manipulator to a given location even if that location is not the desired end goal. This is accomplished by updating the agent's observation and goal information such that it is rewarded as if what it ended up doing was correct all along and as a result the agent improves at performing that action. Another example could be to consider throwing a basketball. Even if a shot is missed, if the shooter considers how the shot was made and how it traveled throughout its journey they can learn how to not only reproduce the shot later should their environment call for it but they can also learn how to adjust in order to hit their original target, a bit of mental gradient descent if you will.  
  
### Proximal Policy Optimization  
  
**Discrete:**  
  
PPO is an on-policy algorithm, this particular implementation computes reward-to-go and action advantage over the course of an entire epoch, however per-episode and even per-reward updates are possible according to the algorithm as described in the paper [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) under section 5 algorithm 1.  
  
> In this experiment a PPO agent was trained in the CartPole-v1 environment for 200 epochs with 12500 steps per epoch, a max of 250 steps per episode. Each epoch ran for approximately 15 minutes, a significant improvement over discrete SAC.  
  
See the graphs below for a glimpse of how things went.  
  
| Discrete PPO Learning Curve | Discrete PPO Reward | CartPole Result |
| ----------- | ----------- | ----------- |
| ![DISC PPO Learning Curve](https://github.com/laughlin-joseph/ProjectAssets/blob/master/CARTPOLE_PPO_DISC/LearningCurve.PNG) | ![DISC PPO Reward](https://github.com/laughlin-joseph/ProjectAssets/blob/master/CARTPOLE_PPO_DISC/Reward.PNG) | ![DISC PPO BEST RESULT](https://github.com/laughlin-joseph/ProjectAssets/blob/master/CARTPOLE_PPO_DISC/RenderSmall.gif) |  
  
PPO performs quite well in this environment, especially when compared with SAC's performance in the same environment, with the same applicable hyperparameters. PPO achieves a higher net reward, learns with more stability and has increasing reward troughs. This learning stability combined with this particular agent's much improved wall time values makes training a PPO Agent an attractive solution for some environments.  
  
So far we've seen that SGD SAC excels at operating in high dimensional continuous observation and action spaces where near-term decisions matter more than future decisions. By contrast, PPO seems to handle more simple problems where starting conditions and subsequent actions all matter in a particular sequence for the sake of achieving a given goal and does so with a higher degree of sample and compute efficiency. At least, the aforementioned feels like a reasonable interpretation of this repository's implementations.  
  
**Continuius**  
  
PPO can be used in both discrete and continuous action environments, in fact, the means by which this is achieved has been adapted to update SAC so that it can operate in discrete environments.  
In discrete environments the actor network outputs logits for each possible discrete action based on a given observation, these odds are then used to create a categorical distribution from which an action is sampled during evaluation time, observations, actions, rewards, Qvals, and action log probabilities are stored in a buffer for later rollout. Trajectory rollouts are then performed to calculate action advantage, which is then used in the clipped PPO objective function to scale the ratio of new and old log probabilities. Advantageous changes have their probabilities increased, harmful changes become less likely, and overall changes are limited by the clip operator. Continuous environments are similar, however instead of calculating logits and log probabilities for all possible actions, the actor net outputs a mean and standard deviation.  These values are used to construct a Gaussian distribution that is used for action sampling, the log probabilities of these actions are calculated during optimization and used accordingly.  

> Here we have the results of running a PPO agent for 200 epochs, 5000 steps per epoch, with 50 steps per episode max against the sparse FetchPickAndPlace-v2 environment.
   
Just below that we have the same test run against the **dense** version of the same environment. Both environments are goal aware, the agents are fed this information by concatenating the observations with their intended goals.  
  
| Continuous Sparse PPO Learning Curve | Continuous Spsrse PPO Histogram | Continuous Sparse PPO Reward | Fetch Pick Place Sparse Result |
| ----------- | ----------- | ----------- | ----------- |
| ![CONT_SPARSE_PPO Learning Curve](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_PPO_SPARSE/LearningCurve.PNG) | ![CONT_SPARSE PPO Histogram](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_PPO_SPARSE/Histo.PNG) | ![CONT_SPARSE PPO Reward](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_PPO_SPARSE/Reward.PNG) | ![CONT_SPARSE PPO BEST RESULT](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_PPO_SPARSE/RenderSmall.gif) |  
  
PPO, in its basic implementation, doesn't handle sparse environments very well.  Reward is nearly absent, learning stops early, and it appears the agent becomes stuck performing very poorly.
  
| Continuous Dense PPO Learning Curve | Continuous Dense PPO Histogram | Continuous Dense PPO Reward | Fetch Pick Place Dense Result |
| ----------- | ----------- | ----------- | ----------- |
| ![CONT_DENSE_PPO Learning Curve](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_PPO_DENSE/LearningCurve.PNG) | ![CONT_DENSE PPO Histogram](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_PPO_DENSE/Histo.PNG) | ![CONT_DENSE PPO Reward](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_PPO_DENSE/Reward.PNG) | ![CONT_DENSE PPO BEST RESULT](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_PPO_DENSE/RenderSmall.gif) |  
  
Yet, with a meaningful reward signal the algorithm performs far better. The histograms paint a picture of increased exploration in action sampling over time when compared to the sparse environment. Further, the agent receives a more meaningful reward signal to aid it in exploration. Nevertheless, SAC with its entropy regularization, temp tuning, and access to HER and PER buffers performs decidedly better.
  
## Conclusions  

Individually, nearly every aspect of machine learning is fascinating. How does a simple linear function, such as a line, learn? Linear algebra solved this issue ages ago with ATAx=ATb, and a single neuron network brought the question into the domain of deep learning. Now we optimize WTx+b, the modern cousin of mx+b, after it is fed through a series of activation functions to end up representing a high dimensionality function, a function approximation beyond our ability to imagine.  From the days of the relatively simple [Neocognitron](https://www.youtube.com/watch?v=KAazjZoiCd0) to the revolutionary publishing of [Attention Is All You Need](https://arxiv.org/abs/1706.03762) one thing is clear, every discovery and advancement depends on that which came before it.  
  
Rainbow DQN, boosting methods, ensembles of traditional statistical models, inception modules, and stacks of networks, all of these show us that individual discoveries can be combined to find an often radically efficient solution to what appears to be an intractable issue. Many RL algorithms still struggle to solve problems that classical control has solved decades ago. However, the idea of a general learner that can be adapted to solve a seemingly endless sea of problems is all too attractive of a prize to rest on past discoveries. In these experiments we've seen how HER, a seemingly counterintuitive idea when applied to an experience, can be added to a very powerful algorithm to aid it in solving a difficult problem of control. 
  
It is my belief that today's existing conundrums can be addressed in a similar fashion. By combining different known solutions with new ideas led by the thinking that gave rise to the solutions we presently have it will be possible to go beyond the limitations of today's technology where new impossible problems will be found. It is only by learning, experimenting, failing, and sharing our results that we can hope to achieve these breakthroughs. To that end, please send any suggestions, bug fixes, corrections, or observations to my email address laughlin dot joseph at gmail dot com. Beyond that, clone this repo, open it, and make something better to share, if you do, kindly mention me in the readme.  
  
### Special thanks  
[Farama Foundation](https://farama.org/)  
[OpenAI](https://github.com/openai)  
[Alexander Van de Kleut](https://github.com/avandekleut)  
[Petros Christodoulou](https://github.com/p-christ)  
[Haibin Zhou et. al.](https://github.com/coldsummerday)  
