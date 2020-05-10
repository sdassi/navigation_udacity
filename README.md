# navigation_udacity
This is the first project from deep reinforcement learning nanodegree of udacity

## Project details

#### Problem definition
This project is about training an agent to navigate in a large square world and collect yellow bananas while avoiding blue ones.

#### State and action spaces
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
* **0** -move forward
* **1** -move backward
* **2** -turn left
* **3** -turn right

#### Reward and score
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 
The score is the sum of rewards obtained during an episode. 

#### Solving the environment
the task is episodic, and the environment is considered solved, when the agent get an average score of +13 over 100 consecutive episodes.

## Getting started
1. You can start by cloning this project `git clone git@github.com:sdassi/navigation_udacity.git`
2. Install all requirements from the requirements file `pip install -r requirements.txt`
3. Download the environment from the link below (select only the one matching your OS):
    * Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    * Mac OSX: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    * Windows (32-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    * Windows (64-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.
5. Place the downloaded file in the location you want (you can place it in this repo for example), then unzip (or decompress the file).

## Instructions
This code allow you to train the agent or evaluate it.
Note that the project already contains a pre-trained weights, if you want to skip the training part and try to evaluate a trained agent that's totally feasible.

#### Train the agent
You can start training the agent with this command: 
```python train_agent.py --file_name "path_Banana_file"```
`train_agent.py` has many arguments. But only `file_name` is essential to specify. It's totally okay if you use default values for the remaining arguments (The way to use default values for aguments is simply not specifying them in the execution command).
This is the list of all arguments that can be passed to `train_agent.py`:
- `file_name`: string argument, path of the banana file (example `Banana_Linux/Banana.x86_64`)
- `n_episodes`: integer argument, maximal number of training episodes, default: 2000
- `max_t`: int argument, maximal number of timesteps during an episode, default: 1000
- `eps_start`: float argument, epsilon starting value, default: 1.0
- `eps_end`: float argument, the lowest value epsilon can reach, default: 0.01
- `eps_decay`: float argument, decreasing factor of epsilon, default: 0.995 

#### Run episode with trained agent 
To evaluate the training agent, you can run:
```python eval_agent.py --file_name "path_Banana_file"```
`eval_agent.py` file has only two arguments:
- `file_name`: The same as defined in the previous section
- `n_episodes`: number of episodes to run, default value 1