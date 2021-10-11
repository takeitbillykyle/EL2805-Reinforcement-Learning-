'''
AUTHORS:
    NORSTRÖM, ARVID 19940206-3193,
    HISELIUS, LEO 9402214192
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
actor = torch.load('models/neural-network-2-actor.pt')
critic = torch.load('models/neural-network-2-critic.pt')

ys = np.linspace(0,1.5,100)
omegas = np.linspace(-np.pi,np.pi,100)



Q = np.zeros((100,100))
for i,y in enumerate(ys):
	for j,omega in enumerate(omegas):
		state = torch.tensor(np.array([0,y,0,0,omega,0,1,1]),dtype=torch.float32).view(1,8)
		
		Q[j,i] = critic(state, torch.tensor(actor(state),dtype = torch.float32).view(1,2)).max(0)[0].detach().numpy()

X,Y = np.meshgrid(ys,omegas)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('y-position')
ax.set_ylabel('Lander angle')
ax.plot_surface(X,Y,Q)
plt.show()




'''
mu_theta = np.zeros((100,100))	
for i,y in enumerate(ys):
	for j,omega in enumerate(omegas):
		state = torch.tensor(np.array([0,y,0,0,omega,0,0,0]),dtype=torch.float32).view(1,8)
		
		mean = actor(state).detach().numpy()
		mu_theta[j,i] = mean[0,1]

X,Y = np.meshgrid(ys,omegas)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,mu_theta)
ax.set_xlabel('y-position')
ax.set_ylabel('Lander angle')
plt.show()
'''
