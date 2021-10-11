'''
AUTHORS:
    NORSTRÖM, ARVID 19940206-3193,
    HISELIUS, LEO 9402214192
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
agent = torch.load('best.pt')

ys = np.linspace(0,1.5,10)
omegas = np.linspace(-np.pi,np.pi,10)

Q = np.zeros((100,100))
zz = []
for i,y in enumerate(ys):
	for j,omega in enumerate(omegas):
		state = torch.tensor(np.array([0,y,0,0,omega,0,1,1]),dtype=torch.float32)
		#Q[i,j] = agent(state).max(0)[0].detach().numpy()
		zz.append([y,omega,agent(state).max(0)[0].detach().numpy()])


zz=np.array(zz)
#X,Y = np.meshgrid(ys,omegas)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(zz[:,0],zz[:,1],zz[:,2])
# ax.matshow(np.array(([1,2,3,4],[1,2,3,4])))
# ax.set_yticks([])
# ax.set_xticks([0,1,2,3])
#ax.plot_surface(X,Y,Q)
#ax.set_xlabel('y')
#ax.set_ylabel('omega')
plt.show()