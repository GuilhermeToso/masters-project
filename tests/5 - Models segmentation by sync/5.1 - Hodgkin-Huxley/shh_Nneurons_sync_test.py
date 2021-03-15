""" 
Stochastic Hodgkin-Huxley Neurons
=================================

Analysis of 12 Neurons coupled in 3 different groups
----------------------------------------------------

**Author**: Guilherme M. Toso
**Tittle**: shh_Nneurons_sunc_test.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization


**Description**:

    This script  uses the Hodgkin-Huxley Biological Neuron Model with Stochastic terms,
    and synchronizes 12 neurons in 3 different groups, such that the neurons in the same group
    are synchronized, while the neurons in different groups are desynchronized. This script plots 
    12 stochastic trajectories, as much as their differences (|V\:sub:`i` - V\:sub:`j`|),
    the growing phases (\phi\:sub:`i`, \phi\:sub:`j`), and the phases difference(|\phi\:sub:`i` - \phi\:sub:`j`|)

"""

""" Dependencies """
import sys
import os
path = os.getcwd()
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

""" Define the total amount of neurons """
neurons = 12

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

""" Cluster Amount """
cluster = 3

""" Determine the coupling force """
k = np.zeros(shape=(neurons,neurons)) + 0.8
print(k)

""" Determine the adjacent matrix that represents which oscillators are coupled
    and in this case, none of them """
adjacency = np.array([\
    [0,0,0,0,1,0,0,0,1,0,0,1],\
    [0,0,1,0,0,1,0,0,0,0,1,0],\
    [0,1,0,0,0,1,0,0,0,0,1,0],\
    [0,0,0,0,0,0,1,1,0,1,0,0],\
    [1,0,0,0,0,0,0,0,1,0,0,1],\
    [0,1,1,0,0,0,0,0,0,0,1,0],\
    [0,0,0,1,0,0,0,1,0,1,0,0],\
    [0,0,0,1,0,0,1,0,0,1,0,0],\
    [1,0,0,0,1,0,0,0,0,0,0,1],\
    [0,0,0,1,0,0,1,1,0,0,0,0],\
    [0,1,1,0,0,1,0,0,0,0,0,0],\
    [1,0,0,0,1,0,0,0,1,0,0,0]])

k = k*adjacency

""" Instantiate the Couple class """
couple = Couple()

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

    next_v = v + sde.membrane_potential(v,m,n,h,current) - ch.synapse(y,v)*step - couple.synapse(k)
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


""" Calculate the Sodium and Potassium Conductances """
gna1 = 120*(data1[1]**3)*data1[3]
gk1 = 36*(data1[2]**4)

red_colors = ["orangered", 'darkred', 'firebrick', 'red']
green_colors = ["limegreen", 'forestgreen', 'darkgreen', 'green']
blue_colors = ["royalblue", "midnightblue","mediumblue", "blue"]

""" Total colors """
colors = red_colors + green_colors + blue_colors

""" Organize the data """
data2 = np.array([data1[0][0], data1[0][4], data1[0][8], data1[0][11],\
    data1[0][1], data1[0][2], data1[0][5], data1[0][10], \
        data1[0][3], data1[0][6], data1[0][7], data1[0][9]])


""" Get the peak indexes, times and the periods between them """
inds, times, pers = unwrap.get_peaks_indexes(data2[:,:].T, 40, T)
#print(inds)

neurons_array = []
for i in range(len(times)):

    neurons_array.append(np.zeros(times[i].size)+i)

cols = ['r','g','b']
labeled = {'Grupo 1':[0,1,2,3], 'Grupo 2':[4,5,6,7], 'Grupo 3':[8,9,10,11]}
ngplot.neural_activity(times, neurons_array,t_max, colors = cols, labeled=labeled)

""" Get the phases """
phases = unwrap.unwrap_static_2(data2.shape[1], inds, T, model='HH')

""" Plot phases """
ngplot.phases(phases, colors, T)


""" Plot the trajectories """

plt.plot(time, data1[0][0], c = red_colors[0])
plt.plot(time, data1[0][4], c = red_colors[1])
plt.plot(time, data1[0][8], c = red_colors[2])
plt.plot(time, data1[0][11], c = red_colors[3])
plt.xlabel('t [ms]',fontsize=34, labelpad=10)
plt.ylabel('V [mV]',fontsize=34, labelpad=30)
plt.yticks(fontsize=34)
plt.xticks(fontsize=34)
plt.grid(True)
plt.title("Neurônios Sincronizados pertencentes ao Grupo 1", fontsize = 24)
plt.show()


plt.plot(time, data1[0][1], c = green_colors[0])
plt.plot(time, data1[0][2], c = green_colors[1])
plt.plot(time, data1[0][5], c = green_colors[2])
plt.plot(time, data1[0][10], c = green_colors[3])
plt.xlabel('t [ms]',fontsize=34, labelpad=10)
plt.ylabel('V [mV]',fontsize=34, labelpad=30)
plt.yticks(fontsize=34)
plt.xticks(fontsize=34)
plt.grid(True)
plt.title("Neurônios Sincronizados pertencentes ao Grupo 2", fontsize = 24)
plt.show()

plt.plot(time, data1[0][3], c = blue_colors[0])
plt.plot(time, data1[0][6], c = blue_colors[1])
plt.plot(time, data1[0][7], c = blue_colors[2])
plt.plot(time, data1[0][9], c = blue_colors[3])
plt.xlabel('t [ms]',fontsize=34, labelpad=10)
plt.ylabel('V [mV]',fontsize=34, labelpad=30)
plt.yticks(fontsize=34)
plt.xticks(fontsize=34)
plt.grid(True)
plt.title("Neurônios Sincronizados pertencentes ao Grupo 3", fontsize = 24)
plt.show()

plt.plot(time, data1[0][11], c = red_colors[3])
plt.plot(time, data1[0][10], c = green_colors[3])
plt.plot(time, data1[0][3], c = blue_colors[0])
plt.plot(time, data1[0][6], c = blue_colors[1])
plt.plot(time, data1[0][7], c = blue_colors[2])
plt.plot(time, data1[0][9], c = blue_colors[3])
plt.xlabel('t [ms]',fontsize=34, labelpad=10)
plt.ylabel('V [mV]',fontsize=34, labelpad=30)
plt.yticks(fontsize=34)
plt.xticks(fontsize=34)
plt.grid(True)
plt.title("Neurônios Dessincronizados dos grupos 1, 2 e 3", fontsize = 24)
plt.show()

""" Get the Phases difference  with group1 as reference"""
ngplot.phases_diff_3D(0, phases, T)

""" Get the Phases difference  with group2 as reference"""
ngplot.phases_diff_3D(4, phases, T)

""" Get the Phases difference  with group2 as reference"""
ngplot.phases_diff_3D(8, phases, T)

""" Get the Trajectories difference  with group1 as reference"""
ngplot.trajecs_diff_3D(0, data2, T)

""" Get the Trajectories difference  with group2 as reference"""
ngplot.trajecs_diff_3D(4, data2, T)

""" Get the Trajectories difference  with group3 as reference"""
ngplot.trajecs_diff_3D(8, data2, T)