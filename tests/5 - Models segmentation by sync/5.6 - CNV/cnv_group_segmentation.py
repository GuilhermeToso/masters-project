""" 
Chaotic Courbage-Nekorkin-Vdovin Neurons
=================================

Analysis of 12 Neurons coupled in 3 different groups
----------------------------------------------------

**Author**: Guilherme M. Toso
**Tittle**: cnv_group_segmentation.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization


**Description**:

    This script  uses the Courbage-Nekorkin-Vdovin Biological Neuron Model with Stochastic terms,
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
from nsc import CNV, Couple
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
dt=1

""" Data Properties """
data = np.zeros((total,2,neurons))

""" Define os x, y e I iniciais """
x = np.random.uniform(0,0.01,size=(neurons))
y = np.zeros((neurons))
t = 0

""" Determine the coupling force """
k = np.zeros(shape=(neurons,neurons)) + 1
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

sigma = 0.2

for s in tqdm(range(time.size)):

    data[s] = np.array([x,y])
    time[s] = t

    couple.data = x

    x = cnv.potential(x,y) - couple.synapse(k)
    y = cnv.recovery(x,y,j)

    t = t + 1


""" Transpose the data array """
data1 = np.transpose(data,(1,2,0))

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
inds, times, pers = unwrap.get_peaks_indexes(data2[:,:].T, 0.2, dt)
print(times)

neurons_array = []
for i in range(len(times)):

    neurons_array.append(np.zeros(times[i].size)+i)

cols = ['r','g','b']
labeled = {'Grupo 1':[0,1,2,3], 'Grupo 2':[4,5,6,7], 'Grupo 3':[8,9,10,11]}
ngplot.neural_activity(times, neurons_array,total, colors = cols, labeled=labeled)

""" Get the phases """
phases = unwrap.unwrap_static_2(total, inds, dt, model='CNV')

""" Plot phases """
ngplot.phases(phases, colors, dt)


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