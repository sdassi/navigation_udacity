import argparse
import gym
import random
import torch
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

from dqn_agent import Agent


def dqn(env, brain_name, n_episodes, max_t, eps_start, eps_end, eps_decay):
    """Deep Q-Learning.
    
    Params
    ======
        env (unityagents.environment.UnityEnvironment): Instance of our environment
        brain_name (str): Name of the env's brain
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for _ in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', default='', type=str)
    parser.add_argument('--n_episodes', default=2000, type=int)
    parser.add_argument('--max_t', default=1000, type=int)
    parser.add_argument('--eps_start', default=1.0, type=float)
    parser.add_argument('--eps_end', default=0.01, type=float)
    parser.add_argument('--eps_decay', default=0.995, type=float)
    args = parser.parse_args()

    #Instantiate the environment
    env = UnityEnvironment(file_name=args.file_name)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    #Instantiate the agent
    state_size=37
    action_size=4
    agent = Agent(state_size, action_size, 0)

    #Train the agent with dqn
    scores = dqn(env, brain_name, args.n_episodes, args.max_t, args.eps_start, args.eps_end, args.eps_decay)

    #Plot scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    #close env
    env.close()