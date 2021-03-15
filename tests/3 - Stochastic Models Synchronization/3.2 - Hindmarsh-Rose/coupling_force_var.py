""" 
Stochastic Hindmarsh-Rose Neurons
=================================

Coupling Force Variation
------------------------

**Author**: Guilherme M. Toso
**Tittle**: coupling_force_var.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization

**Description**:

    This script uses the Hindmarsh-Rose Biological Neuron Model with Stochastic terms,
    it uses two neurons and then try to synchronize them by varying yhe coupling force k. This script plots 
    the differences (|V\:sub:`i` - V\:sub:`j`|) and the phases difference(|\phi\:sub:`i` - \phi\:sub:`j`|) of the two trajectories
    of every k value.

"""


import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import HindmarshRose, Couple, ngplot, unwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from tqdm import tqdm

np.random.seed(0)

""" Instantiate Hindmarsh and Rose Class """
hr = HindmarshRose(1,3.0,1.0,5.0,0.001,1.6,-1.6,1)

""" Neurons amount """
neurons = 2

""" Time properties """
t,t_f,dt = 0.0,50,0.01
iters = int(t_f/dt)

""" Coupling Force Vector """
k = np.linspace(0,1.92,num=100)

def force_variation(k, neurons, decimals = 2):

    num = k.size
    k = np.repeat(k,repeats=neurons**2)
    k = np.reshape(k,(num,neurons,neurons))
    k[:,np.arange(neurons), np.arange(neurons)] = 0
    k = np.around(k,decimals=decimals)

    return k
force = force_variation(k,neurons)

""" Instantiate the Couple class """
couple = Couple()

""" Create the data structure to store the trajectories differences and the phases """
diff_data = np.zeros((k.size, iters))
phases_data = np.zeros((k.size, iters))

sigma = 1.0

for i in tqdm(range(k.size)):

    t = 0
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

        couple.data = X

        next_X = X + hr.potential(X,Y,Z)*dt + sigma*X*np.random.uniform(0,dt,size=X.size) - couple.synapse(force[i])
        next_Y = Y + hr.fast_ion_channel(X,Y)*dt + sigma*X*np.random.uniform(0,dt,size=X.size)
        next_Z = Z + hr.slow_ion_channel(X,Z)*dt + sigma*X*np.random.uniform(0,dt,size=X.size)
        
        init[0,:] = next_X
        init[1,:] = next_Y
        init[2,:] = next_Z

        t = t + dt

    
    """ Store the trajecs difference data with the ith element of coupling force k """
    diff_data[i] = np.abs(data[:,0,0] - data[:,0,1])

    """ Get the peak indexes, times and the periods between them """
    inds, times, pers = unwrap.get_peaks_indexes(data[:,0,:], 1, dt)
    """ Get the phases """
    phases = unwrap.unwrap_static_2(iters, inds, dt,model='HR')

    """ Store the phases difference data with the ith element of coupling force k """
    phases_data[i] = np.abs(phases[0] - phases[1])

ngplot.coupling(diff_data, phases_data, k, time)