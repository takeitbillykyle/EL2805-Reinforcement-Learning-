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
from DDPG_agent import RandomAgent, Critic, Actor
from DDPG_agent import ExperienceReplayBuffer
import torch.optim as optim
import torch.nn as nn




class LunarLander():
    env=gym.make('LunarLanderContinuous-v2')
    env.reset()
    #"cuda:0" if torch.cuda.is_available() else
    ddev = "cpu"
    dev = torch.device(ddev)
    n_ep_running_average = 50
    def __init__(self, gamma, L, N_episodes, N, net_dims):

        # HYPERPARAMETERS
        self.gamma = gamma
        self.L = L
        self.N_episodes = N_episodes
        self.N = N
        self.C = int(self.L/self.N_episodes)
        self.C = 2
        self.tau = 1e-3

        # MISC
        self.episode_reward_list = []
        self.episode_number_of_steps = []
        self.EPISODES = trange(self.N_episodes, desc='Episode: ', leave=True)
        self.Experience = namedtuple('Experience',
                            ['state', 'action', 'reward', 'next_state', 'done'])
        self.buffert = self.init_buffert(ExperienceReplayBuffer(self.L),self.L)
        self.best_reward_run_av = np.NINF
        self.prev_noise = np.zeros((2,))

        # FOR TRAINING
        self.net_dims = net_dims
        self.main_critic = Critic().to(self.ddev)
        self.target_critic = Critic().to(self.ddev)
        self.main_actor = Actor(self.net_dims).to(self.ddev)
        self.target_actor = Actor(self.net_dims).to(self.ddev)

        self.optimizer_critic = optim.Adam(self.main_critic.parameters(), lr=5e-4)
        self.optimizer_actor = optim.Adam(self.main_actor.parameters(), lr=5e-5)
        self.loss_critic = nn.MSELoss()
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
    def UOnoise(self, mu = 0.15, sigma = 0.2):
        return -mu*self.prev_noise+np.random.multivariate_normal([0,0],sigma**2*np.eye(2), size = 1)
    def train(self, PATH = 'models/best.pt', SAVE=False):
        for i in self.EPISODES:
            # Reset enviroment data and initialize variables
            done = False
            state = self.env.reset()
            total_episode_reward = 0.
            t = 0
            self.prev_noise = np.array([0,0])
            while not done:
                UOnoise = self.UOnoise()

                with torch.no_grad():
                    action = self.main_actor(torch.tensor([state], device=self.dev,requires_grad = False, dtype = torch.float32))
                self.prev_noise = UOnoise
             
                # Get next state and reward.  The done variable
                # will be True if you reached the goal position,
                # False otherwise

                action = action.detach().numpy().reshape(2,)+UOnoise.reshape(2,)
                action = action.clip(-1,1)
                next_state, reward, done, x_ = self.env.step(action)

                exp = self.Experience(state, action, reward, next_state, done)

                self.buffert.append(exp)

                states, actions, rewards, next_states, dones = self.buffert.sample_batch(self.N)

                # Training process, set gradients to 0
                self.optimizer_critic.zero_grad()

                # Compute output of the network given the states batch
               
                Q_pred_values = self.main_critic(torch.tensor(states, device=self.dev, requires_grad = True,  dtype=torch.float32),
                    torch.tensor(actions, device=self.dev, requires_grad = False, dtype = torch.float32).detach()).view(self.N,1)
                
                Q_target = self.target_critic(torch.tensor(next_states,device=self.dev, requires_grad = False, dtype=torch.float32).detach(), 
                    torch.tensor(self.target_actor(torch.tensor(next_states,device=self.dev, requires_grad = False, dtype = torch.float32).detach()), requires_grad = False, dtype = torch.float32).detach()).max(1)[0]
                #print(dones)
                Q_target_values = (torch.tensor(rewards, device=self.dev,dtype=torch.float32)+self.gamma*Q_target*(1-torch.tensor(list(map(int,dones)),device=self.dev))).view(self.N,1)
                # Compute loss function
                self.main_critic.train(mode=True)
                loss_critic = self.loss_critic(Q_pred_values,Q_target_values)

                # Compute gradient
                loss_critic.backward()

                # Clip gradient norm to 1
                nn.utils.clip_grad_norm_(self.main_critic.parameters(), max_norm=1.)

                # Perform backward pass (backpropagation)
                self.optimizer_critic.step()

                if t%self.C == 0:
                    self.optimizer_actor.zero_grad()
                    self.main_actor.train(mode = True)
                    loss_actor = -torch.mean(self.main_critic(torch.tensor(states, device=self.dev,requires_grad = False, dtype = torch.float32).detach(),
                        self.main_actor(torch.tensor(states, device=self.dev,requires_grad = True, dtype = torch.float32))))
                    loss_actor.backward()
                    #print(loss_actor)
                    nn.utils.clip_grad_norm(self.main_actor.parameters(), max_norm=1.)
                    self.optimizer_actor.step()
                    tgt_state = self.target_actor.state_dict()
                    for k, v in self.main_actor.state_dict().items():
                        tgt_state[k] = (1 - self.tau)*tgt_state[k] + self.tau*v
                    self.target_actor.load_state_dict(tgt_state)

                    tgt_state = self.target_critic.state_dict()
                    for k,v in self.main_critic.state_dict().items():
                        tgt_state[k] = (1 - self.tau)*tgt_state[k] + self.tau*v
                    self.target_critic.load_state_dict(tgt_state)


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
            if SAVE:
                if reward_run_av > self.best_reward_run_av:
                    self.best_reward_run_av = reward_run_av
                    torch.save(self.main_critic, 'models/best_critic.pt')
                    torch.save(self.main_actor, 'models/best_actor.pt')
                
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
        plt.savefig(str(self.gamma)+'_'+str(self.L)+'_'+str(self.N_episodes)+'_'+str(self.N)+'_'+str(self.net_dims)+'.png')
        #plt.show()

    def init_buffert(self,buffert,L):
        filled = False
        counter = 0
        while not filled:
            done = False
            state = self.env.reset()
            while not done:
                action = np.clip(-1 + 2 * np.random.rand(2), -1, 1)
                next_state,reward,done,_ = self.env.step(action)
                exp = self.Experience(state, action, reward, next_state, done)
                buffert.append(exp)
                counter +=1
                if counter == L:
                    filled = True
                    break
        return buffert


ll = LunarLander(0.99, 60000, 500, 64, [8,400,200   ,2])
ll.train()
ll.plot()
