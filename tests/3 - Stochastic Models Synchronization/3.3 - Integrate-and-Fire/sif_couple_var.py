""" 
Stochastic Integrate-and-Fire Neurons
=================================

Coupling Force Variation
------------------------

**Author**: Guilherme M. Toso
**Tittle**: sif_couple_var.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization

**Description**:

    This script uses the Integrate-and-Fire Biological Neuron Model with Stochastic terms,
    it uses two neurons and then try to synchronize them by varying yhe coupling force k. This script plots 
    the differences (|V\:sub:`i` - V\:sub:`j`|) and the phases difference(|\phi\:sub:`i` - \phi\:sub:`j`|) of the two trajectories
    of every k value.

"""

import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import IntegrateAndFire, Couple, ngplot, unwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from tqdm import tqdm

#np.random.seed(0)

""" Integrate and Fire Parameters """
vrest = 0.0
r = 1.0
tau = 10.0
threshold = 1.0
I = 2.5

""" Instantiates the Integrate and Fire Model Class """
IF = IntegrateAndFire(vrest,r,tau,threshold)

""" Neurons amount """
neurons = 2
""" Time Properties """
total = 200

""" Coupling Force Vector """
k = np.linspace(0,1.2,num=100)
def force_variation(k, neurons, decimals = 2):

    num = k.size
    k = np.repeat(k,repeats=neurons**2)
    k = np.reshape(k,(num,neurons,neurons))
    k[:,np.arange(neurons), np.arange(neurons)] = 0
    k = np.around(k,decimals=decimals)

    return k
force = force_variation(k,neurons)

couple = Couple()

""" Create the data structure to store the trajectories differences and the phases """
diff_data = np.zeros((k.size, total+1))
phases_data = np.zeros((k.size, total+1))

sigma = 0.3

for i in tqdm(range(k.size)):

        
    time = np.linspace(0,total,(total+1))

    """ Data array """
    data = np.zeros((total+1,neurons))

    u = np.random.uniform(0,0.5,size=neurons)


    for j in tqdm(range(time.size)):

        data[j] = u

        couple.data = u

        next_u = u + IF.lif(u,I) + sigma*u*np.random.normal(0,0.2,size=neurons) - couple.synapse(force[i])

        u = IF.reset(data[j],next_u)

    """ Store the trajecs difference data with the ith element of coupling force k """
    diff_data[i] = np.abs(data[:,0] - data[:,1])

    """ Get the peak indexes, times and the periods between them """
    inds, times, pers = unwrap.get_peaks_indexes(data[:,:], threshold, 1)
    """ Get the phases """
    phases = unwrap.unwrap_static_2(total+1, inds, 1,model='IAF')

    """ Store the phases difference data with the ith element of coupling force k """
    phases_data[i] = np.abs(phases[0] - phases[1])

ngplot.coupling(diff_data, phases_data, k, time)
