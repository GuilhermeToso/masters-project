""" 
Chaotic Izhikevich Neurons
=================================

Coupling Force Variation
------------------------

**Author**: Guilherme M. Toso
**Tittle**: izhikevich_force_var.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization

**Description**:

    This script uses the Izhikevich Biological Neuron Model with Stochastic terms,
    it uses two neurons and then try to synchronize them by varying yhe coupling force k. This script plots 
    the differences (|V\:sub:`i` - V\:sub:`j`|) and the phases difference(|\phi\:sub:`i` - \phi\:sub:`j`|) of the two trajectories
    of every k value.

"""


import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import Izhikevic, Couple, ngplot, unwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from tqdm import tqdm

#np.random.seed(0)

""" Neurons Amount """
neurons = 2

""" Izhikevic Parameters """
a = 0.02
b = 0.2
c = -55
d = 0.93
thresh = -40

""" Instantiate Izhikevic Model """
iz = Izhikevic(a,b,c,d)

""" Time Properties """
t = 0
amount = 400
total = amount + 1
time = np.zeros((total))
step = 1

""" Coupling Force Vector """
f = np.linspace(0.01,1.5,num=100)
def force_variation(k, neurons, decimals = 2):

    num = k.size
    k = np.repeat(k,repeats=neurons**2)
    k = np.reshape(k,(num,neurons,neurons))
    k[:,np.arange(neurons), np.arange(neurons)] = 0
    k = np.around(k,decimals=decimals)

    return k
force = force_variation(f,neurons)
print(force)
""" Instantiate Couple class """
couple = Couple()

""" Create the data structure to store the trajectories differences and the phases """
diff_data = np.zeros((f.size, total))
phases_data = np.zeros((f.size, total))

for i in tqdm(range(f.size)):

    time = np.zeros(shape=(total))

    """ Data Properties """
    data = np.zeros((total,2,neurons))

    """ Define os x, y e I iniciais """
    v = np.random.uniform(-65,-64,size=neurons)

    u = b*v
    I = np.zeros((neurons)) + 10
    t = 0

    for j in tqdm(range(time.size)):

        data[j] = np.array([v,u])
        time[j] = t


        couple.data = v

        v1 = v + iz.potential(v,u,I)*step - couple.synapse(force[i])
        u1 = u + iz.recovery(v,u)*step

        new = iz.update(v1,u1)

        v, u = new[0], new[1]

        t = t + step

    """ Store the trajecs difference data with the ith element of coupling force k """
    diff_data[i] = np.abs(data[:,0,0] - data[:,0,1])

    """ Get the peak indexes, times and the periods between them """
    inds, times, pers = unwrap.get_peaks_indexes(data[:,0,:], thresh, 0.5)
    """ Get the phases """
    phases = unwrap.unwrap_static_2(total, inds, 0.5,model='Izhikevic')

    """ Store the phases difference data with the ith element of coupling force k """
    phases_data[i] = np.abs(phases[0] - phases[1])

ngplot.coupling(diff_data, phases_data, f, time)