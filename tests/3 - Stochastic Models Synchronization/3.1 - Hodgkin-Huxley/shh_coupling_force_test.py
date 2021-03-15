""" 
Stochastic Hodgkin-Huxley Neurons
=================================

Coupling Force Variation
------------------------

**Author**: Guilherme M. Toso
**Tittle**: shh_coupling_force_test.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization

**Description**:

    This script uses the Hodgkin-Huxley Biological Neuron Model with Stochastic terms,
    it uses two neurons and then try to synchronize them by varying yhe coupling force k from 0 to 1, with step 0.01. This script plots 
    the differences (|V\:sub:`i` - V\:sub:`j`|) and the phases difference(|\phi\:sub:`i` - \phi\:sub:`j`|) of the two trajectories
    of every k value.

"""

""" Dependencies """
import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import HodgkinHuxley, Chemical, SDE, Couple
from nsc import unwrap
from nsc import ngplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from tqdm import tqdm
import sys

np.random.seed(0)

""" Define the total amount of neurons """
neurons = 2

""" Define the initial parameters """
v_na = np.zeros(neurons) + 115
v_k = np.zeros(neurons) - 12
v_l = np.zeros(neurons) + 86
g_na = np.zeros(neurons) + 120
g_k = np.zeros(neurons) + 36
g_l = np.zeros(neurons) + .3
C = np.ones(neurons)


""" Frequency """    
freq = 50.0

""" Period """
T = 1/freq

""" Step value """
step = T

""" Total time """
t_max = 300.0

""" Iterations """
n_iter = int(t_max/step)

""" Stochasticity of the model """
sigma = 1.
sigma_ext = 0.5

""" Instantiate the Hodgkin-Huxley Class """
hh = HodgkinHuxley(v_na, v_k, v_l, g_na, g_k, g_l, C)
sde = SDE(step,sigma_ext, sigma, v_na, v_k, v_l, g_na, g_k, g_l, C)

""" Define the Chemical class params and it's instatiation """
jch = 0.1
v_reversal = -70
ch = Chemical(jch,v_reversal)


""" Create the array of time """
time = np.zeros((n_iter))


""" Coupling force vector """
k = np.linspace(0,1,num=100)

def force_variation(k, neurons, decimals = 2):

    num = k.size
    k = np.repeat(k,repeats=neurons**2)
    k = np.reshape(k,(num,neurons,neurons))
    k[:,np.arange(neurons), np.arange(neurons)] = 0
    k = np.around(k,decimals=decimals)

    return k

force = force_variation(k,neurons)

""" Create the data structure to store the trajectories differences and the phases """

diff_data = np.zeros((k.size, n_iter))
phases_data = np.zeros((k.size, n_iter))

couple = Couple()

""" For evey value of k """
for i in tqdm(range(k.size)):

    """ Define the initial time t, and the variables V, M, N, and H """

    t = 0.0
    v = np.random.uniform(0.0, 4.0, (neurons))
    m = hh.m0(v)[2]
    n = hh.n0(v)[2]
    h = hh.h0(v)[2]
    y = ch.y0(v)[2]

    """ Define the array where will be stored all Variables (V, M, N and H) of all Neurons at all times. """
    data = np.zeros((n_iter,5,neurons))

    """ Initialize the matrix init that contains all Variables of all Neurons at time t """
    init = np.array([v,m,n,h,y])


    """ Begin the Iteration Process """
    for j in range(n_iter):
        
        """ Stores the matrix init at the data array in the time i """
        data[j] = init
        
        """ The array time at iteration i receives the value of t """
        time[j] = t
        
        """ Define the initial Variables """
        v, m, n, h, y = init[0], init[1], init[2], init[3], init[4]
        
        couple.data = v

        """ Set the electrical current I """
        current = 20
        
        next_v = v + sde.membrane_potential(v,m,n,h,current) - ch.synapse(y,v)*step - couple.synapse(force[i])
        next_m = m + sde.stochastic_sodium_activate(m,v)
        next_h = h + sde.stochastic_sodium_deactivate(h,v)
        next_n = n + sde.stochastic_potassium_activate(n,v)
        next_y = y + sde.stochastic_chemical_transmitter(y,v)

        init[0] = next_v
        init[1] = next_m
        init[2] = next_n
        init[3] = next_h
        init[4] = next_y

        
        """ Update Time """
        t = t + step



    """ Transpose the data array """
    data1 = np.transpose(data,(1,2,0))

    """ Store the trajecs difference data with the ith element of coupling force k """
    diff_data[i] = np.abs(data1[0][0] - data1[0][1])

    """ Get the peak indexes, times and the periods between them """
    inds, times, pers = unwrap.get_peaks_indexes(data[:,0,:], 40, T)
 
    """ Get the phases """
    phases = unwrap.unwrap_static_2(data.shape[0], inds, T,model='HH')
    
    """ Store the phases difference data with the ith element of coupling force k """
    phases_data[i] = np.abs(phases[0] - phases[1])

ngplot.coupling(diff_data, phases_data, k, time)
