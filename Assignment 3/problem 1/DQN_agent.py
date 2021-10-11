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

class Agent(nn.Module):
    ''' Base agent class, used as a parent class
        Args:
            n_actions (int): number of actions
        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''

    def __init__(self, layers):

        super(Agent, self).__init__()
        
        no_layers = len(layers)
        modules = []
        for i in range(no_layers-2):
            modules.append(nn.Linear(layers[i],layers[i+1]))
            modules.append(nn.PReLU())
        modules.append(nn.Linear(layers[-2],layers[-1]))
        self.model = nn.Sequential(*modules)


    def forward(self, state):
        ''' Performs a forward computation '''
        state = self.model(state)
        return state


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices
            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action