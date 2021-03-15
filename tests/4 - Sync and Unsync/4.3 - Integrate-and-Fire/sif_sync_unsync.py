""" 
Stochastic Integate-And-Fire Neurons
=================================

Analysis of Synchronization and Unsync through time
----------------------------------------------------

**Author**: Guilherme M. Toso
**Tittle**: sif_sync_unsync.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization


**Description**:

    This script  uses the Integate-And-Fire Biological Neuron Model with Stochastic terms,
    and synchronizes 2 neurons in 1 group, then after a certain time we will unsync these neurons.

"""

""" Dependencies """
import sys
import os
path = os.getcwd() + '\\Neurons_Synchronization_Competition'
sys.path.insert(0,path)
from nsc import IntegrateAndFire, Couple
from nsc import unwrap
from nsc import ngplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from tqdm import tqdm

np.random.seed(0)

""" Neurons """
neurons = 2
""" Integrate and Fire Parameters """
vrest = 0.0
r = 1.0
tau = 10.0
threshold = 1.0
I = 2.5

""" Instantiates the Integrate and Fire Model Class """
IF = IntegrateAndFire(vrest,r,tau,threshold)

""" Time Properties """
total = 200
dt = 1
sigma=0.2
time = np.linspace(0,total,(total+1))

""" Data array """
data = np.zeros((total+1,neurons))

u = np.random.uniform(0,0.5,size=neurons)

k = 0
adjacency = np.array(
    [[0,1],[1,0]]
)

couple = Couple()

def update_force(k, i, x, iters=total):

    t_1 = int(0.25*iters)
    t_3 = int(0.75*iters)


    if i <= t_1:
        x = x + 0.01
        k = 0.4/(1+np.exp(-5*(x-100)))
        print("t_1: ", k)
    elif i >= t_3:
        x = x + 0.01
        k = 0.4*np.exp(-1.5*x)
        print("t_3: ", k)
    elif t_1 < i < t_3:
        k = 0.8
        x = 0

    return k, x 
x = 0


for j in tqdm(range(time.size)):

    data[j] = u

    couple.data = u

    next_u = u + IF.lif(u,I) + sigma*u*np.random.normal(0,0.2,size=neurons) - couple.synapse(k*adjacency)

    u = IF.reset(data[j],next_u)

    k,x = update_force(k,j,x)

""" Transpose the data array """
data1 = data.T

""" Store the trajecs difference data with the ith element of coupling force k """
diff_data = np.abs(data1[0] - data1[1])

""" Get the indexes, times of the neurons' fires and
    periods between each one """
inds, times, pers = unwrap.get_peaks_indexes(data[:,:], 1, 1)

""" Calculate the phases """
phases = unwrap.unwrap_static_2(len(data), inds, 1, model='IAF')

""" Colors to plot the phases """
colors = ['b','r']


""" Plot Trajectories """
ngplot.trajectories(data[:,:], time)

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