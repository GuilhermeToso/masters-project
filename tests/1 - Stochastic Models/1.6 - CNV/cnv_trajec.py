""" 
Chaotic Courbage-Nekorkin-Vdovin Neuron
================================


**Author**: Guilherme M. Toso
**Tittle**: cnv_trajecs.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization

**Description**:

    Analysis of the Chaotic Courbage-Nekorkin-Vdovin Neuron Model

"""

import sys
import os
import platform
path = os.getcwd()
sys.path.insert(0,path)
from nsc import CNV
from nsc import unwrap
from nsc import ngplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from tqdm import tqdm

np.random.seed(0)

neurons = 1
neurons = 2
neurons = 100

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

    x = cnv.potential(x,y)
    y = cnv.recovery(x,y,j)

    t = t + 1
    
""" Store the trajecs difference data with the ith element of coupling force k """
data1 = np.transpose(data,(1,2,0))

""" Get the peak indexes, times and the periods between them """
inds, times, pers = unwrap.get_peaks_indexes(data[:,0,:], 0.2, 1)
""" Get the phases """
phases = unwrap.unwrap_static_2(total, inds, 1,model='CNV')

""" For neurons = 2 or 100, if is 1, comment this piece of code """
neurons_array = []
for i in range(len(times)):
    neurons_array.append(np.zeros(times[i].size)+i)
ngplot.neural_activity(times, neurons_array,t)


""" Colors to plot the phases """

""" Plasma colormap """
plasma = cm.get_cmap('plasma',neurons)
color = list(plasma.colors) # for 100 neurons
#color = ['b','r']           # for 2 neurons 
#color = ['b']

""" Plot the phases """
ngplot.phases(phases,color, 1)

""" Plot the phases diff """
#ngplot.phases_diff(0, phases, 1, colors=['b'], set_limit=True) # For neurons = 2
ngplot.phases_diff_3D(0, phases, 1)   # For neurons = 100, for neurons = 1, comment both lines

ngplot.trajectories(data[:,0,:],time,colors=color)

""" Plot the trajectories difference if neurons = 2 or 100"""
for i in range(neurons-1):
    i = i + 1
    plt.plot(time, np.abs(data1[0][0] - data1[0][i]), c = color[i])
plt.xlabel('t [ms]',fontsize=34, labelpad=10)
plt.ylabel(r"$|V_{i} - V_{j}|$ [mV]",fontsize=34, labelpad=30)
plt.yticks(fontsize=34)
plt.xticks(fontsize=34)
plt.grid(True)
plt.show()
