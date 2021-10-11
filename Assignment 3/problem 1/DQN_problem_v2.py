'''
AUTHORS:
    NORSTRÖM, ARVID 19940206-3193,
    HISELIUS, LEO 9402214192
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




class LunarLander():
    env = gym.make('LunarLander-v2')
    env.reset()

    n_ep_running_average = 50
    def __init__(self, gamma, L, N_episodes, N, net_dims):

        # HYPERPARAMETERS
        self.gamma = gamma
        self.L = L
        self.N_episodes = N_episodes
        self.N = N
        self.C = int(self.L/self.N)

        # MISC
        self.episode_reward_list = []
        self.episode_number_of_steps = []
        self.EPISODES = trange(self.N_episodes, desc='Episode: ', leave=True)
        self.Experience = namedtuple('Experience',
                            ['state', 'action', 'reward', 'next_state', 'done'])
        self.buffert = self.init_buffert(ExperienceReplayBuffer(self.L),self.N)
        self.best_reward_run_av = np.NINF
        
        # FOR TRAINING
        self.net_dims = net_dims
        self.main_agent = Agent(self.net_dims)
        self.target_agent = Agent(self.net_dims)
        self.optimizer = optim.Adam(self.main_agent.parameters(), lr=4e-4)
        self.loss_function = nn.MSELoss()
    def running_average(self,x, N):
        ''' Function used to compute the running average
            of the last N elements of a vector x
        '''
        if len(x) >= N:
            y = np.copy(x)
            y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
        else:
            y = np.zeros_like(x)
        return y

    def train(self, PATH = 'models/best_cyclic2.pt'):
        eps_min = 0.05
        eps_max = 0.95
        Z = 0.95*self.N_episodes
        for i in self.EPISODES:
            # Reset enviroment data and initialize variables
            done = False
            state = self.env.reset()
            total_episode_reward = 0.
            t = 0

            while not done:
                # Take epsilon greedy action

                #epsilon = np.max((eps_min, eps_max*(eps_min/eps_max)**(i/Z)))
                epsilon = np.min([0.9999,np.max([0.0001,np.exp(-t/200)*np.abs(np.cos(2*np.pi*t/300))])])
                sample = np.random.binomial(1, 1-epsilon)
                state_tensor = torch.tensor([state],
                                            requires_grad=False,
                                            dtype=torch.float32)
                values = self.main_agent(state_tensor)
                action = np.random.randint(4) if sample == 0 else values.max(1)[1].item() 
             
                # Get next state and reward.  The done variable
                # will be True if you reached the goal position,
                # False otherwise
                next_state, reward, done, _ = self.env.step(action)
                self.env.render()
                exp = self.Experience(state, action, reward, next_state, done)

                self.buffert.append(exp)

                states, actions, rewards, next_states, dones = self.buffert.sample_batch(self.N)

                # Training process, set gradients to 0
                self.optimizer.zero_grad()

                # Compute output of the network given the states batch
                Q_pred_values = self.main_agent(torch.tensor(states, requires_grad = True,  dtype=torch.float32)).gather(1, torch.tensor(actions).view(self.N,1)) #n
                
                Q_target = self.target_agent(torch.tensor(next_states, dtype=torch.float32)).detach().max(1)[0]
                #print(dones)
                Q_target_values = (torch.tensor(rewards, dtype=torch.float32)+self.gamma*Q_target*(1-torch.tensor(list(map(int,dones))))).view(self.N,1)
                # Compute loss function
                self.main_agent.train(mode=True)
                loss = self.loss_function(Q_pred_values,Q_target_values)

                # Compute gradient
                loss.backward()

                # Clip gradient norm to 1
                nn.utils.clip_grad_norm_(self.main_agent.parameters(), max_norm=1.)

                # Perform backward pass (backpropagation)
                self.optimizer.step()

                if t%self.C == 0:
                    self.target_agent.load_state_dict(self.main_agent.state_dict())


                # Update episode reward
                total_episode_reward += reward

                # Update state for next iteration
                state = next_state
                t += 1

            # Append episode reward and total number of steps
            self.episode_reward_list.append(total_episode_reward)
            self.episode_number_of_steps.append(t)

            # Close environment
            self.env.close()

            # Updates the tqdm update bar with fresh information
            # (episode number, total reward of the last episode, total number of Steps
            # of the last episode, average reward, average number of steps)
            reward_run_av = self.running_average(self.episode_reward_list, self.n_ep_running_average)[-1]
            steps_run_av = self.running_average(self.episode_number_of_steps, self.n_ep_running_average)[-1]

            self.EPISODES.set_description("Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
                i, total_episode_reward, t,
                reward_run_av,
                steps_run_av))

            #save model
            #if reward_run_av > self.best_reward_run_av:
            #    self.best_reward_run_av = reward_run_av
            #    torch.save(self.main_agent, PATH)
                
    def plot(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
        ax[0].plot([i for i in range(1, self.N_episodes + 1)], self.episode_reward_list, label='Episode reward')
        ax[0].plot([i for i in range(1, self.N_episodes + 1)], self.running_average(
            self.episode_reward_list, self.n_ep_running_average), label='Avg. episode reward')
        ax[0].set_xlabel('Episodes')
        ax[0].set_ylabel('Total reward')
        ax[0].set_title('Total Reward vs Episodes')
        ax[0].legend()
        ax[0].grid(alpha=0.3)

        ax[1].plot([i for i in range(1, self.N_episodes + 1)], self.episode_number_of_steps, label='Steps per episode')
        ax[1].plot([i for i in range(1, self.N_episodes + 1)], self.running_average(
            self.episode_number_of_steps, self.n_ep_running_average), label='Avg. number of steps per episode')
        ax[1].set_xlabel('Episodes')
        ax[1].set_ylabel('Total number of steps')
        ax[1].set_title('Total number of steps vs Episodes')
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        #plt.savefig(str(self.gamma)+'_'+str(self.L)+'_'+str(self.N_episodes)+'_'+str(self.N)+'_'+str(self.net_dims)+'.png')
        plt.show()

    def init_buffert(self,buffert,L):
        filled = False
        counter = 0
        while not filled:
            done = False
            state = self.env.reset()
            while not done:
                action = np.random.randint(4)
                next_state,reward,done,_ = self.env.step(action)
                exp = self.Experience(state, action, reward, next_state, done)
                buffert.append(exp)
                counter +=1
                if counter == L:
                    filled = True
                    break
        return buffert



ll = LunarLander(0.99, 30000, 4000, 64, [8,64,4])
ll.train()
