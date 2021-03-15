""" 
Chaotic Courbage-Nekorkin-Vdovin Neurons
=================================

Coupling Force Variation
------------------------

**Author**: Guilherme M. Toso
**Tittle**: cnv_force_var.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization

**Description**:

    This script uses the Courbage-Nekorkin-Vdovin Biological Neuron Model with Stochastic terms,
    it uses two neurons and then try to synchronize them by varying yhe coupling force k. This script plots 
    the differences (|V\:sub:`i` - V\:sub:`j`|) and the phases difference(|\phi\:sub:`i` - \phi\:sub:`j`|) of the two trajectories
    of every k value.

"""


import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import CNV, Couple, ngplot, unwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from tqdm import tqdm

np.random.seed(0)


""" Neurons Amount """
neurons = 2

""" Define os parâmetros """
a = 0.002
m0 = 0.4
m1 = 0.3
d = 0.3
e = 0.01
j = 0.1123
beta = 0.1

""" Instancia um objeto do modelo CNV """
cnv = CNV(a, m0, m1, d, beta, e)


""" Time Properties """
t = 0.0
amount = 2500
total = amount + 1
time = np.zeros((total))


""" Coupling Force Vector """
f = np.linspace(0.1,1,num=100)
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
diff_data = np.zeros((f.size, total))
phases_data = np.zeros((f.size, total))

for i in tqdm(range(f.size)):

    time = np.zeros(shape=(total))

    """ Data Properties """
    data = np.zeros((total,2,neurons))

    """ Define os x, y e I iniciais """
    x = np.random.uniform(0,0.01,size=(neurons))
    y = np.zeros((neurons))
    t = 0

    for k in tqdm(range(time.size)):

        data[k] = np.array([x,y])
        time[k] = t

        couple.data = x

        x = cnv.potential(x,y) - couple.synapse(force[i])
        y = cnv.recovery(x,y,j)
    
        t = t + 1
    
    """ Store the trajecs difference data with the ith element of coupling force k """
    diff_data[i] = np.abs(data[:,0,0] - data[:,0,1])
    """ Get the peak indexes, times and the periods between them """
    inds, times, pers = unwrap.get_peaks_indexes(data[:,0,:], 0.2, 1)
    """ Get the phases """
    phases = unwrap.unwrap_static_2(total, inds, 1,model='CNV')

    """ Store the phases difference data with the ith element of coupling force k """
    phases_data[i] = np.abs(phases[0] - phases[1])

ngplot.coupling(diff_data, phases_data, f, time)