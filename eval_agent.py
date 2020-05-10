import argparse
import torch
from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt

from dqn_agent import Agent

def run_episode(env, brain_name, max_timesteps=1000):
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    eps = 0.01 #even if we're not training, small eps can avoid getting stuck
    for _ in range(max_timesteps):
        action = agent.act(state, eps)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        state = next_state
        score += reward
        if done:
            break
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', default='', type=str)
    parser.add_argument('--n_episodes', '-n', default=1, type=int)
    args = parser.parse_args()

    env = UnityEnvironment(file_name=args.file_name)

    brain_name = env.brain_names[0]

    state_size=37
    action_size=4
    agent = Agent(state_size, action_size, 0)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    scores = []
    for i in range(args.n_episodes):
        score = run_episode(env, brain_name)
        scores.append(score)
    print("Average scores :  ", np.mean(scores))