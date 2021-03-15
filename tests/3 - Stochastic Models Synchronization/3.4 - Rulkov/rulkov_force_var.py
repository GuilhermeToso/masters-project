""" 
Chaotic Rulkov Neurons
=================================

Coupling Force Variation
------------------------

**Author**: Guilherme M. Toso
**Tittle**: rulkov_force_var.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization

**Description**:

    This script uses the Rulkov Biological Neuron Model with Stochastic terms,
    it uses two neurons and then try to synchronize them by varying yhe coupling force k. This script plots 
    the differences (|V\:sub:`i` - V\:sub:`j`|) and the phases difference(|\phi\:sub:`i` - \phi\:sub:`j`|) of the two trajectories
    of every k value.

"""



import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import Rulkov, Couple, ngplot, unwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from tqdm import tqdm

""" Neurons Amount """
neurons = 2

""" Rulkov Parameters """
a = 4.2
alpha = 0.001
sigma = -1.2


""" Instantiate Rulkov Model """
rk = Rulkov(a,alpha,sigma)


""" Time Properties """
step = 1
t = 2000
time = np.linspace(0,t,(t+1))

""" Coupling Force Vector """
f = np.linspace(0.01,0.5,num=100)
def force_variation(k, neurons, decimals = 2):

    num = k.size
    k = np.repeat(k,repeats=neurons**2)
    k = np.reshape(k,(num,neurons,neurons))
    k[:,np.arange(neurons), np.arange(neurons)] = 0
    k = np.around(k,decimals=decimals)

    return k
force = force_variation(f,neurons)

""" Instantiate the Couple class """
couple = Couple()

""" Create the data structure to store the trajectories differences and the phases """
diff_data = np.zeros((f.size, t+1))
phases_data = np.zeros((f.size, t+1))

for i in tqdm(range(f.size)):

    time = np.linspace(0,t,(t+1))

    """ Data Properties """
    data = np.zeros((t+1,2,neurons))

    """ Define os x, y e I iniciais """
    x = np.random.uniform(-1,-1.2,size=neurons)
    y = np.zeros((neurons)) - 2.9
    I = np.zeros((neurons))

    for j in tqdm(range(time.size)):

        couple.data = x

        fast = rk.fx(x,y,I) - couple.synapse(force[i])
        slow = rk.fy(x,y)

        x = fast
        y = slow

        data[j] = np.array([x,y])

    """ Store the trajecs difference data with the ith element of coupling force k """
    diff_data[i] = np.abs(data[:,0,0] - data[:,0,1])

    """ Get the peak indexes, times and the periods between them """
    inds, times, pers = unwrap.get_peaks_indexes(data[:,0,:], -1.7, 1)
    """ Get the phases """
    phases = unwrap.unwrap_static_2(t+1, inds, 1, model='Rulkov')

    """ Store the phases difference data with the ith element of coupling force k """
    phases_data[i] = np.abs(phases[0] - phases[1])


ngplot.coupling(diff_data, phases_data, f, time)