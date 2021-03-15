""" 
Stochastic Hindmarsh-Rose Neuron
================================


**Author**: Guilherme M. Toso
**Tittle**: shr_trajecs.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization

**Description**:

    Analysis of the Stochastic Hindmarsh-Rose Neuron Model

"""


import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import HindmarshRose
from nsc import unwrap
from nsc import ngplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from tqdm import tqdm

np.random.seed(0)

#neurons = 1
#neurons = 2
neurons = 100

""" Instantiate Hindmarsh and Rose Class """
hr = HindmarshRose(1,3.0,1.0,5.0,0.001,1.6,-1.6,1)

""" Time properties """
t,t_f,dt = 0.0,50,0.01
iters = int(t_f/dt)

sigma = 1.0

""" Initialize x, y and z """
init = np.random.uniform(-0.5,.5,(3,neurons))
""" Create the time and x,y, and z tensors """
time = np.zeros((iters))
data = np.zeros((iters,3,neurons))

for j in tqdm(range(iters)):

    data[j] = init
    time[j] = t

    X = init[0,:]
    Y = init[1,:]
    Z = init[2,:]

    next_X = X + hr.potential(X,Y,Z)*dt + sigma*X*np.random.uniform(0,dt,size=X.size)
    next_Y = Y + hr.fast_ion_channel(X,Y)*dt + sigma*X*np.random.uniform(0,dt,size=X.size)
    next_Z = Z + hr.slow_ion_channel(X,Z)*dt + sigma*X*np.random.uniform(0,dt,size=X.size)
    
    init[0,:] = next_X
    init[1,:] = next_Y
    init[2,:] = next_Z

    t = t + dt
""" Transpose the data array """
data1 = np.transpose(data,(1,2,0))

""" Get the peak indexes, times and the periods between them """
inds, times, pers = unwrap.get_peaks_indexes(data[:,0,:], 1, dt)
""" Get the phases """
phases = unwrap.unwrap_static_2(iters, inds, dt,model='HR')


""" For neurons = 2 or 100, if is 1, comment this piece of code """
neurons_array = []
for i in range(len(times)):
    neurons_array.append(np.zeros(times[i].size)+i)
ngplot.neural_activity(times, neurons_array,t_f)


""" Colors to plot the phases """

""" Plasma colormap """
plasma = cm.get_cmap('plasma',neurons)
color = list(plasma.colors) # for 100 neurons
#color = ['b','r']           # for 2 neurons 
#color = ['b']               # for 1 neuron

""" Plot the phases """
ngplot.phases(phases,color, dt)

""" Plot the phases diff """
#ngplot.phases_diff(0,phases,dt,colors=['b']) # For neurons = 2
ngplot.phases_diff_3D(0, phases, dt)       # For neurons = 100, for neurons = 1, comment both lines


ngplot.trajectories(data[:,0],time,colors=color)

""" Plot the trajectories difference if neurons = 2 or 100"""
for i in range(neurons-1):
    i = i+1
    plt.plot(time,np.abs(data1[0][0] - data1[0][i]), c=color[i])
plt.xlabel('t [ms]',fontsize=34, labelpad=10)
plt.ylabel(r"$|V_{i} - V_{j}|$ [mV]",fontsize=34, labelpad=30)
plt.yticks(fontsize=34)
plt.xticks(fontsize=34)
plt.grid(True)
plt.show()
