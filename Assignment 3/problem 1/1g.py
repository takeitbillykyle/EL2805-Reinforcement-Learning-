'''
AUTHORS:
    NORSTRÖM, ARVID 19940206-3193,
    HISELIUS, LEO 19940221-4192
'''
from collections import namedtuple
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent, Agent
from DQN_agent import ExperienceReplayBuffer
import torch.optim as optim
import torch.nn as nn
from DQN_problem_v2 import LunarLander

best_ll = LunarLander(0.99, 3000, 300, 64, [8,64,64,4])
best_ll.main_agent = torch.load('models/best.pt')

episodes = np.arange(50)
env = gym.make('LunarLander-v2')

episode_reward_list = []
for i in episodes:
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    while not done:
        # Take epsilon greedy action
        state_tensor = torch.tensor([state],
                                    requires_grad=False,
                                    dtype=torch.float32)
        action = np.argmax(best_ll.main_agent.forward(state_tensor).detach().cpu().numpy()[0])
        #action = np.random.randint(0,4)

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)

        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t += 1

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)

    # Close environment
    env.close()

np.save("best_agent", episode_reward_list)