# Computer lab 0

# Problem 1: Shortest Path in the Maze

import numpy as np
import matplotlib.pyplot as plt


X = np.arange(6)
Y = np.arange(7)

states = [[x,y] for x in X for y in Y]
obstacles = [(0,2), (1,2), (2,2), (4,1), (4,2), (4,3), (4,4), (4,5)]
goal = (5,5)


def reward(state):
    """
    Args: 
        state: tuple (i,j) corresponding to grid-position
    Returns:
        the reward

    """
    
    if state in obstacles:
        return -99
    elif state == goal:
        return 0
    else:
        return -1

def prob(state,action):
    """
    Args:
        State and action given as tuples
    Returns:
        the probability of success
    
    """
    state_prime = tuple(np.array(action)+np.array(state))
    if state_prime in states and state_prime not in obstacles:
        return 1
    else:
        return 0


actions = [(0,1), (1,0), (0,-1), (-1,0), (0,0)]
actions_str = {(0,1) : 'right', (1,0) : 'down', (0,-1) : 'left', (-1,0) : 'up', (0,0) : 'stay'}
V = np.zeros((6,7))
for _ in range(3):
    for state in states:
        v = V[state]
        candidates = []
        for action in actions:
            state_prime = tuple(np.array(action)+np.array(state))
            if state_prime in states:
                candidates.append(prob(state,action)*(reward(state)+V[state_prime]))

        V[state] =np.max(candidates)
        V[goal] = 0
cell_text = np.array(np.zeros((6,7)), dtype = str)

for state in states:
    candidates = {}
    for action in actions:
        state_prime = tuple(np.array(action)+np.array(state))
        if state_prime in states:
            candidates[action] = V[state_prime]
    if state in obstacles:
        best_action = 'OBSTACLE'
    else:
        best_action = actions_str[max(candidates,key=candidates.get)]

    cell_text[state] = best_action
#cell_text = np.array((100*V-(100*V)%10)/100,dtype=str)
print(V)
plt.table(cellText=cell_text,loc='center')
plt.show()