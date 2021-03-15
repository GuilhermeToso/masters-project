""" 
Stochastic Integrate-and-Fire Neurons
=================================

Analysis of 12 Neurons coupled in 3 different groups
----------------------------------------------------

**Author**: Guilherme M. Toso
**Tittle**: sif_group_segmentation.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization


**Description**:

    This script  uses the Integrate-and-Fire Biological Neuron Model with Stochastic terms,
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
from nsc import IntegrateAndFire, Couple
from nsc import unwrap
from nsc import ngplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import sys
from tqdm import tqdm
np.random.seed(0)

""" Neurons """
neurons = 12
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

time = np.linspace(0,total,(total+1))

""" Data array """
data = np.zeros((total+1,neurons))

u = np.random.uniform(0,0.5,size=neurons)


""" Determine the coupling force """
k = np.zeros(shape=(neurons,neurons)) + 1.2
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

sigma = 0.1

for j in tqdm(range(time.size)):

    data[j] = u

    couple.data = u

    next_u = u + IF.lif(u,I) + sigma*u*np.random.normal(0,0.2,size=neurons) - couple.synapse(k)

    u = IF.reset(data[j],next_u)

""" Transpose the data array """
data1 = np.transpose(data,(1,0))

red_colors = ["orangered", 'darkred', 'firebrick', 'red']
green_colors = ["limegreen", 'forestgreen', 'darkgreen', 'green']
blue_colors = ["royalblue", "midnightblue","mediumblue", "blue"]

""" Total colors """
colors = red_colors + green_colors + blue_colors

""" Organize the data """
data2 = np.array([data1[0], data1[4], data1[8], data1[11],\
    data1[1], data1[2], data1[5], data1[10], \
        data1[3], data1[6], data1[7], data1[9]])


""" Get the peak indexes, times and the periods between them """
inds, times, pers = unwrap.get_peaks_indexes(data2[:,:].T, threshold, dt)


neurons_array = []
for i in range(len(times)):

    neurons_array.append(np.zeros(times[i].size)+i)

cols = ['r','g','b']
labeled = {'Grupo 1':[0,1,2,3], 'Grupo 2':[4,5,6,7], 'Grupo 3':[8,9,10,11]}
ngplot.neural_activity(times, neurons_array,total, colors = cols, labeled=labeled)

""" Get the phases """
phases = unwrap.unwrap_static_2(total, inds, dt, model='IAF')

""" Plot phases """
ngplot.phases(phases, colors, dt)


""" Plot the trajectories """

plt.plot(time, data1[0], c = red_colors[0])
plt.plot(time, data1[4], c = red_colors[1])
plt.plot(time, data1[8], c = red_colors[2])
plt.plot(time, data1[11], c = red_colors[3])
plt.xlabel('t [ms]',fontsize=34, labelpad=10)
plt.ylabel('V [mV]',fontsize=34, labelpad=30)
plt.yticks(fontsize=34)
plt.xticks(fontsize=34)
plt.grid(True)
plt.title("Neurônios Sincronizados pertencentes ao Grupo 1", fontsize = 24)
plt.show()


plt.plot(time, data1[1], c = green_colors[0])
plt.plot(time, data1[2], c = green_colors[1])
plt.plot(time, data1[5], c = green_colors[2])
plt.plot(time, data1[10], c = green_colors[3])
plt.xlabel('t [ms]',fontsize=34, labelpad=10)
plt.ylabel('V [mV]',fontsize=34, labelpad=30)
plt.yticks(fontsize=34)
plt.xticks(fontsize=34)
plt.grid(True)
plt.title("Neurônios Sincronizados pertencentes ao Grupo 2", fontsize = 24)
plt.show()

plt.plot(time, data1[3], c = blue_colors[0])
plt.plot(time, data1[6], c = blue_colors[1])
plt.plot(time, data1[7], c = blue_colors[2])
plt.plot(time, data1[9], c = blue_colors[3])
plt.xlabel('t [ms]',fontsize=34, labelpad=10)
plt.ylabel('V [mV]',fontsize=34, labelpad=30)
plt.yticks(fontsize=34)
plt.xticks(fontsize=34)
plt.grid(True)
plt.title("Neurônios Sincronizados pertencentes ao Grupo 3", fontsize = 24)
plt.show()

plt.plot(time, data1[11], c = red_colors[3])
plt.plot(time, data1[10], c = green_colors[3])
plt.plot(time, data1[3], c = blue_colors[0])
plt.plot(time, data1[6], c = blue_colors[1])
plt.plot(time, data1[7], c = blue_colors[2])
plt.plot(time, data1[9], c = blue_colors[3])
plt.xlabel('t [ms]',fontsize=34, labelpad=10)
plt.ylabel('V [mV]',fontsize=34, labelpad=30)
plt.yticks(fontsize=34)
plt.xticks(fontsize=34)
plt.grid(True)
plt.title("Neurônios Dessincronizados dos grupos 1, 2 e 3", fontsize = 24)
plt.show()

""" Get the Phases difference  with group1 as reference"""
ngplot.phases_diff_3D(0, phases, dt)

""" Get the Phases difference  with group2 as reference"""
ngplot.phases_diff_3D(4, phases, dt)

""" Get the Phases difference  with group2 as reference"""
ngplot.phases_diff_3D(8, phases, dt)

""" Get the Trajectories difference  with group1 as reference"""
ngplot.trajecs_diff_3D(0, data2, dt)

""" Get the Trajectories difference  with group2 as reference"""
ngplot.trajecs_diff_3D(4, data2, dt)

""" Get the Trajectories difference  with group3 as reference"""
ngplot.trajecs_diff_3D(8, data2, dt)