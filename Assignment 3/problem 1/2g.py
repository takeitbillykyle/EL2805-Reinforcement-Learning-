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
from DDPG_problem import LunarLander

best_ll = LunarLander(0.99, 3000, 300, 64, [8,64,64,4])
best_ll.main_actor = torch.load('models/neural-network-2-actor.pt')

episodes = np.arange(50)
env = gym.make('LunarLanderContinuous-v2')

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

        
        with torch.no_grad():
            action = best_ll.main_actor(torch.tensor([state], device=best_ll.dev,requires_grad = False, dtype = torch.float32))

        action = action.detach().numpy().reshape(2,)
        action = action.clip(-1,1)
        

        #action = np.random.uniform(-1,1,2)
        next_state, reward, done, x_ = env.step(action)

        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t += 1

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)

    # Close environment
    env.close()

np.save("p2_best_agent", episode_reward_list)