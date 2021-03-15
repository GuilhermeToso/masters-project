""" 
Stochastic Hodgkin-Huxley Neurons
=================================

Analysis of two non-coupled neurons
-----------------------------------

**Author**: Guilherme M. Toso
**Tittle**: shh_2neurons_test.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization

**Description**:

    This script  uses the Hodgkin-Huxley Biological Neuron Model with Stochastic terms,
    and plots two stochastic trajectories, as much as their differences (|V\:sub:`1` - V\:sub:`2`|),
    the growing phases (\phi\:sub:`1`, \phi\:sub:`2`), and the phases difference(|\phi\:sub:`1` - \phi\:sub:`2`|)

"""

import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import HodgkinHuxley, SDE, Chemical
from nsc import unwrap
from nsc import ngplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors

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
c = np.ones(neurons)
sigma = 1.0
sigma_external = 3.0

""" Frequency """    
freq = 50.0

""" Period """
T = 1/freq

""" Define the step value for every step """
step = T


""" Instantiate the Hodgkin-Huxley Class """
hh = HodgkinHuxley(v_na, v_k, v_l, g_na, g_k, g_l, c)
sde = SDE(step,sigma_external, sigma, v_na, v_k, v_l, g_na, g_k, g_l, c)

""" Define the chemical initial params and the class """
jch = 0.1
v_reversal = -70
ch = Chemical(jch, v_reversal)

""" Define the total time """
t_max = 300.0

""" Define the number of iterations """
n_iter = int(t_max/step)


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

""" Create the array of time """
time = np.zeros((n_iter))

""" Begin the Iteration Process """
for i in range(len(time)):
    

    """ Stores the matrix init at the data array in the time i """
    data[i] = init
    
    """ The array time at iteration i receives the value of t """
    time[i] = t
    
    """ Define the initial Variables """
    v = init[0]
    m = init[1]
    n = init[2]
    h = init[3]
    y = init[4]
    
    """ Set the electrical current I """
    current = 20
    
    next_v = v + sde.membrane_potential(v,m,n,h,current) - ch.synapse(y,v)*step
    next_m = m + sde.stochastic_sodium_activate(m,v)
    next_n = n + sde.stochastic_potassium_activate(n,v)
    next_h = h + sde.stochastic_sodium_deactivate(h,v)
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

""" Calculate the Sodium and Potassium Conductances """
gna1 = 120*(data1[1]**3)*data1[3]
gk1 = 36*(data1[2]**4)

""" Colors to plot the phases """
colors= ['b', 'r']

""" Get the indexes, times of the neurons' fires and
    periods between each one """
inds, times, pers = unwrap.get_peaks_indexes(data[:,0,:], 40, T)

""" Calculate the phases """
phases = unwrap.unwrap_static_2(data.shape[0], inds, T,model='HH')

neurons_array = []
for i in range(len(times)):

    neurons_array.append(np.zeros(times[i].size)+i)

ngplot.neural_activity(times, neurons_array,t_max)


""" Plot the phases and the phases difference """
ngplot.phases(phases, ['b','r'], T)
ngplot.phases_diff(0, phases, T, colors=['b'], set_limit=True)

""" Plot the trajectories """
plt.plot(time, data1[0][0], 'b', label = "Neurônio 1")
plt.plot(time, data1[0][1], 'r', label = "Neurônio 2")
plt.xlabel('t [ms]',fontsize=36, labelpad=10)
plt.ylabel('V [mV]',fontsize=36, labelpad=30)
plt.yticks(fontsize=36)
plt.xticks(fontsize=36)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=36)
plt.title("Potenciais do Modelo Estocástico de Hodgkin-Huxley", fontsize = 34)
plt.show()


""" Plot the trajectories difference """
plt.plot(time,np.abs(data1[0][0] - data1[0][1]), 'b', label = r"$|V_{0} - V_{1}|$")
plt.xlabel('t [ms]',fontsize=36, labelpad=10)
plt.ylabel(r"$|V_{i} - V_{j}|$ [mV]",fontsize=36, labelpad=30)
plt.yticks(fontsize=36)
plt.xticks(fontsize=36)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=36)
plt.title("Diferença dos Potenciais do Modelo Estocástico de Hodgkin-Huxley", fontsize = 34)
plt.show()

