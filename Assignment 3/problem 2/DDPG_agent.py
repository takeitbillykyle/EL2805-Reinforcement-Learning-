'''
AUTHORS:
    NORSTRÖM, ARVID 19940206-3193,
    HISELIUS, LEO 9402214192
'''


from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """
    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length) # deque allows me to insert from both ends

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def __len__(self):
        # override len operator
        return len(self.buffer)

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """

        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        indices = np.random.choice(
            len(self.buffer),
            size=n,
            replace=False
        )

        batch = [self.buffer[i] for i in indices]

        return zip(*batch)

class Actor(nn.Module):
    ''' Actor class for DDPG to model policy pi
        Input:
            state s
        Output:
            action a
    '''
    def __init__(self, layers):
        super(Actor, self).__init__()

        no_layers = len(layers)
        modules = []

        for i in range(no_layers -2):
            modules.append(nn.Linear(layers[i],layers[i+1]))
            modules.append(nn.PReLU())
        modules.append(nn.Linear(layers[-2],layers[-1]))
        modules.append(nn.Tanh()) 
        self.model = nn.Sequential(*modules)

    def forward(self, state):
         state = self.model(state)
         return state

class Critic(nn.Module):
    ''' Critic class for DDPG to model policy pi
        Input:
            state s, action a
        Output:
            value v
    '''
    def __init__(self):
        super(Critic, self).__init__()
        action_dim = 2
        self.input_state_layer = nn.Linear(8, 400)   
        self.input_activation = nn.PReLU()
        self.hidden_layer = nn.Linear(400+action_dim, 200)
        self.hidden_activation = nn.PReLU()
        self.output_layer = nn.Linear(200, 1)

    def forward(self, state, action):
        h_state = self.input_state_layer(state)
        h_state = self.input_activation(h_state)
        data_concat = torch.cat([h_state, action], dim = 1)
        penultimate = self.hidden_layer(data_concat)
        penultimate = self.hidden_activation(penultimate)
        out = self.output_layer(penultimate)

        return out



class RandomAgent(object):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)