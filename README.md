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
  
### Soft Actor Critic  
**Discrete:**  
Though initially designed for continuous environments this repo contains an implementation capable of handling them.  

**Continuius**  
This is where the SAC algorithm shines.  
![Learning curve without HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_NO_HER/SACSparseNoHerNoLearn.PNG?raw=true) 
![Reward without HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_NO_HER/SACNoHERRew.PNG?raw=true)  
  
Compared to:  
  
![Learning curve with HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_CONT_SPARSE/FPP_20_Epc_LC_HER_SPARSE.PNG?raw=true)
![Reward with HER](https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_CONT_SPARSE/FPPReward.PNG?raw=true)  

Results:  
<table>
<thead>
<tr>
<th>Without HER</th>
<th>With HER</th>
</tr>
</thead>
<tbody>
<tr>
<td>
<video src="https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_NO_HER/StruggleSmall.gif?raw=true" controls></video>
</td>
<td>
<video src="https://github.com/laughlin-joseph/ProjectAssets/blob/master/FPP_SAC_CONT_SPARSE/WorkingSmall.gif?raw=true" controls></video>
</td>
</tr>
</tbody>
</table>  
  
### Proximal Policy Optimization  
**Discrete:**  
  
**Continuius**  
  
### Special thanks  
[Farama Foundation](https://farama.org/)  
[OpeAI](https://github.com/openai)  
[Alexander Van de Kleut](https://github.com/avandekleut)  
[Petros Christodoulou](https://github.com/p-christ)  
[Haibin Zhou et. al.](https://github.com/coldsummerday)  
