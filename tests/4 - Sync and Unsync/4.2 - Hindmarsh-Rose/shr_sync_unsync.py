""" 
Stochastic Hindmarsh-Rose Neurons
=================================

Analysis of Synchronization and Unsync through time
----------------------------------------------------

**Author**: Guilherme M. Toso
**Tittle**: shr_sync_unsync.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization


**Description**:

    This script  uses the Hindmarsh-Rose Biological Neuron Model with Stochastic terms,
    and synchronizes 2 neurons in 1 group, then after a certain time we will unsync these neurons.

"""

""" Dependencies """
import sys
import os
path = os.getcwd() + '\\Neurons_Synchronization_Competition'
sys.path.insert(0,path)
from nsc import HindmarshRose, Couple
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
neurons = 2

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

k = 0
adjacent = np.array(
    [[0,1],[1,0]]
)

couple = Couple()

def update_force(k, i, x, iters=iters):

    t_1 = int(0.25*iters)
    t_3 = int(0.75*iters)


    if i <= t_1:
        x = x + 0.01
        k = 0.08/(1+np.exp(-5*(x-100)))
        print("t_1: ", k)
    elif i >= t_3:
        x = x + 0.01
        k = 0.08*np.exp(-1.5*x)
        print("t_3: ", k)
    elif t_1 < i < t_3:
        k = 0.1
        x = 0

    return k, x 
x = 0


for j in tqdm(range(iters)):

    data[j] = init
    time[j] = t

    X = init[0,:]
    Y = init[1,:]
    Z = init[2,:]

    couple.data = X
    
    next_X = X + hr.potential(X,Y,Z)*dt + sigma*X*np.random.uniform(0,dt,size=X.size) - couple.synapse(k*adjacent)
    next_Y = Y + hr.fast_ion_channel(X,Y)*dt + sigma*X*np.random.uniform(0,dt,size=X.size)
    next_Z = Z + hr.slow_ion_channel(X,Z)*dt + sigma*X*np.random.uniform(0,dt,size=X.size)
    
    init[0,:] = next_X
    init[1,:] = next_Y
    init[2,:] = next_Z

    t = t + dt

    k,x = update_force(k,j,x)
""" Transpose the data array """
data1 = np.transpose(data,(1,2,0))

""" Store the trajecs difference data with the ith element of coupling force k """
diff_data = np.abs(data1[0][0] - data1[0][1])

""" Get the indexes, times of the neurons' fires and
    periods between each one """
inds, times, pers = unwrap.get_peaks_indexes(data[:,0,:], 1, dt)

""" Calculate the phases """
phases = unwrap.unwrap_static_2(len(data), inds, dt, model='HR')

""" Colors to plot the phases """
colors = ['b','r']


""" Plot Trajectories """
ngplot.trajectories(data[:,0,:], time)

""" Plot the phases """
ngplot.phases(phases, colors, dt)

ngplot.phases_diff(0,phases,dt,colors=['b'])

plt.plot(time, diff_data, 'b')
plt.xlabel('t [ms]', fontsize=34, labelpad=30)
plt.ylabel(r'$|V_{i}-V_{j}|$[mV]', fontsize=34, labelpad=30)
plt.xticks(fontsize=34)
plt.yticks(fontsize=34)
plt.grid()
plt.show()