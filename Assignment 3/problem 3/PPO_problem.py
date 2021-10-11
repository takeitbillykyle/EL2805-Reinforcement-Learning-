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
from PPO_agent import RandomAgent, Critic, Actor
from PPO_agent import ExperienceReplayBuffer
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable



class LunarLander():
    env=gym.make('LunarLanderContinuous-v2')
    env.reset()
    #"cuda:0" if torch.cuda.is_available() else
    ddev = "cpu"
    dev = torch.device(ddev)
    n_ep_running_average = 50
    def __init__(self, gamma, N_episodes, M, dims_critic,dims_actor):

        # HYPERPARAMETERS
        self.gamma = gamma
        self.L = 1000
        self.N_episodes = N_episodes
        self.M = M    
        self.epsilon = 0.9
        # MISC
        self.episode_reward_list = []
        self.episode_number_of_steps = []
        self.EPISODES = trange(self.N_episodes, desc='Episode: ', leave=True)
        self.Experience = namedtuple('Experience',
                            ['state', 'action', 'reward', 'next_state', 'done'])
        self.best_reward_run_av = 100

        # FOR TRAINING
        self.dims_critic = dims_critic
        self.dims_actor = dims_actor
        self.Actor = Actor(self.dims_actor).to(self.ddev)
        self.Critic = Critic(self.dims_critic).to(self.ddev)

        self.optimizer_critic = optim.Adam(self.Critic.parameters(), lr=1e-3)
        self.optimizer_actor = optim.Adam(self.Actor.parameters(), lr=1e-5)
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
    def normpdf(self, action, mean, cov):
        return 1/np.sqrt((2*np.pi)**2*np.linalg.det(cov))*np.exp(-0.5*(action-mean).T@np.linalg.inv(cov)@(action-mean))
    def normpdf_tensor(self, action, mean, cov):
        return 1/torch.sqrt((2*np.pi)**2*torch.linalg.det(cov))*torch.exp(-0.5*(action-mean).T@torch.inverse(cov)@(action-mean))
    def normpdf_1dtensor(self,data,means,variances):
        return 1/torch.sqrt(2*np.pi*variances)*torch.exp((-(data-means)**2/(2*variances)))
    def train(self, SAVE=False):
        for i in self.EPISODES:
            # Reset enviroment data and initialize variables
            done = False
            state = self.env.reset()
            total_episode_reward = 0.
            self.buffert = ExperienceReplayBuffer(self.L)
            t = 0
            tt = 0
            pi_old = []
            while not done:
                with torch.no_grad():
                   
                    action = self.Actor(torch.tensor([state],device = self.dev, requires_grad = False, dtype = torch.float32)).detach().numpy().reshape(2,2)
           
                    mean,cov = action[0].reshape(2,1),np.diag(action[1])
                    action = np.random.multivariate_normal(mean.reshape(2,),cov).reshape(2,1)
                    
                    pi_old.append(torch.tensor(self.normpdf(action,mean,cov)))

                next_state, reward, done, _ = self.env.step(action.reshape(2,))

                exp = self.Experience(state, action, reward, next_state, done)

                self.buffert.append(exp)

                # Update episode reward
                total_episode_reward += reward

                # Update state for next iteration
                state = next_state
                t += 1

           
            rewards = torch.tensor(np.array(self.buffert.buffer)[:,2].astype(float),dtype = torch.float32)
            a = len(rewards)
            pi_old = torch.tensor(pi_old,dtype=torch.float32).view(a,1)
            actions = torch.tensor(np.stack(np.array(self.buffert.buffer)[:,1],axis = 0), dtype = torch.float32).view(a,2)
            states = torch.tensor(np.stack(np.array(self.buffert.buffer)[:,0],axis = 0),requires_grad = True, dtype = torch.float32)
    

            with torch.no_grad():
               G = torch.tensor([self.gamma**torch.arange(0,a-i)@rewards[i:] for i in range(len(rewards))],requires_grad = False,dtype=torch.float32).view(len(rewards),1).detach()


            C = self.Critic(states).view(a,1)
            with torch.no_grad():
                psi = G-C
            for n in range(self.M):
                
                C = self.Critic(states).view(a,1)

                torch.autograd.set_detect_anomaly(True)
                self.optimizer_critic.zero_grad()
   


                loss_critic = self.loss_critic(C,G)
                loss_critic.backward()
                nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=1)
                self.optimizer_critic.step() 


                s=self.Actor(states)

                means,covs = s[0:a], s[a:]

                loss_actor = 0
                #print(actions[:,0].shape, means[:,0].shape, covs[:,0].shape)
                pi=self.normpdf_1dtensor(actions[:,0].view(a,1), means[:,0].view(a,1), covs[:,0].view(a,1))*self.normpdf_1dtensor(actions[:,1].view(a,1), means[:,1].view(a,1), covs[:,1].view(a,1))
                
                r_theta = (pi*1/pi_old).float()
            
                c_epsilon_min = torch.min(torch.cat([r_theta, torch.ones_like(r_theta)+self.epsilon],dim=1),dim=1)[0].view(a,1)
                c_epsilon = torch.max(torch.cat([torch.ones_like(r_theta)-self.epsilon, c_epsilon_min], dim = 1),dim=1)[0].view(a,1)


             

                loss_actor = -torch.mean(torch.min(torch.cat([r_theta*psi,c_epsilon*psi],dim=1),dim=1)[0])

                
                #self.normpdf_1dtensor(actions[0,0], means[0,0], covs[0,0])
                """
                loss_actor = 0
                for j in range(a):

                    pi = self.normpdf_tensor(torch.tensor(actions[j],dtype=torch.float32).view(2,1), means[j].view(2,1), torch.diag(covs[j]))

                    r_theta = (pi/pi_old[j]).float()

                    
                    c_epsilon = max(1-self.epsilon, min(r_theta, 1+self.epsilon))
                 
                    loss_actor -= min(r_theta*psi[j],c_epsilon*psi[j])/a
                """
               
                self.optimizer_actor.zero_grad()
                loss_actor.backward()        
           
                
                nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm =1)
            
                self.optimizer_actor.step()
            self.buffert = ExperienceReplayBuffer(self.L)
          
           

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

            tt+=1

            if SAVE:
                if reward_run_av > self.best_reward_run_av:
                    self.best_reward_run_av = reward_run_av
                    torch.save(self.Critic, 'models/best_critic.pt')
                    torch.save(self.Actor, 'models/best_actor.pt')
                
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
        plt.savefig(str(self.gamma)+'_'+str(self.epsilon)+'.png')
        #plt.show()



ll = LunarLander(0.99, 1600, 10, [8,400,200   ,1],[8,400,200,2])
ll.train(False)
ll.plot()

