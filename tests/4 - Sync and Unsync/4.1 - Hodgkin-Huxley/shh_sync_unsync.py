""" 
Stochastic Hodgkin-Huxley Neurons
=================================

Analysis of Synchronization and Unsync through time
----------------------------------------------------

**Author**: Guilherme M. Toso
**Tittle**: shh_sync_unsync.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization


**Description**:

    This script  uses the Hodgkin-Huxley Biological Neuron Model with Stochastic terms,
    and synchronizes 2 neurons in 1 group, then after a certain time we will unsync these neurons.

"""

""" Dependencies """
import sys
import os
path = os.getcwd() + '\\Neurons_Synchronization_Competition'
print(path)

sys.path.insert(0,path)
from nsc import HodgkinHuxley, SDE, Chemical, Couple
from nsc import unwrap
from nsc import ngplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import sys
np.random.seed(0)


""" neurons """
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
sigma_external = 0.5

""" Frequency """    
freq = 50.0

""" Period """
T = 1/freq

""" Define the step value for every step """
step = T

""" Instantiate the Hodgkin-Huxley Class """
hh = HodgkinHuxley(v_na, v_k, v_l, g_na, g_k, g_l, c)
sde = SDE(step, sigma_external, sigma, v_na, v_k, v_l, g_na, g_k, g_l, c)

""" Define the Chemical class params an instantiate it """
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

""" 

As the Coupling Force varies from 0 to 1, and:

 - Unsync at 0 to 0.1
 - Sync at 0.1 to 1.0

We will test this variation through time, atarting at 0

"""

k = 0
adjacent = np.array(
    [[0,1],[1,0]]
)

""" Instantiate the Couple class """
couple = Couple()

def update_force(k, i, x, iters=n_iter):

    t_1 = int(0.25*iters)
    t_3 = int(0.75*iters)


    if i <= t_1:
        x = x + 0.01
        k = 0.5/(1+np.exp(-5*(x-100)))
    elif i >= t_3:
        x = x + 0.01
        k = 0.5*np.exp(-1.5*x)
    elif t_1 < i < t_3:
        k = 0.5
        x = 0

    return k, x 
x = 0.0
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

    couple.data = v

    next_v = v + sde.membrane_potential(v,m,n,h,current) - ch.synapse(y,v)*step - couple.synapse(k*adjacent)
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

    k, x = update_force(k,i,x)


""" Transpose the data array """
data1 = np.transpose(data,(1,2,0))


""" Calculate the Sodium and Potassium Conductances """
gna1 = 120*(data1[1]**3)*data1[3]
gk1 = 36*(data1[2]**4)

""" Store the trajecs difference data with the ith element of coupling force k """
diff_data = np.abs(data1[0][0] - data1[0][1])

""" Get the indexes, times of the neurons' fires and
    periods between each one """
inds, times, pers = unwrap.get_peaks_indexes(data[:,0,:], 40, T)

""" Calculate the phases """
phases = unwrap.unwrap_static_2(len(data), inds, T, model='HH')

""" Colors to plot the phases """
colors = ['b','r']


""" Plot Trajectories """
ngplot.trajectories(data[:,0,:], time)

""" Plot the phases """
ngplot.phases(phases, colors, T)

ngplot.phases_diff(0,phases,T,colors=['b'])

plt.plot(time, diff_data, 'b')
plt.xlabel('t [ms]', fontsize=34, labelpad=30)
plt.ylabel(r'$|V_{i}-V_{j}|$[mV]', fontsize=34, labelpad=30)
plt.xticks(fontsize=34)
plt.yticks(fontsize=34)
plt.grid()
plt.show()