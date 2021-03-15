""" 
Chaotic Izhikevich Neuron
================================


**Author**: Guilherme M. Toso
**Tittle**: iz_trajecs.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization

**Description**:

    Analysis of the Chaotic Izhikevich Neuron Model

"""

import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import Izhikevic
from nsc import unwrap
from nsc import ngplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from tqdm import tqdm

np.random.seed(0)

#neurons = 1
#neurons = 2
neurons = 100

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
amount = 300
total = amount + 1
time = np.zeros((total))
step = 1

time = np.zeros(shape=(total))

""" Data Properties """
data = np.zeros((total,2,neurons))

""" Define os x, y e I iniciais """
v = np.random.uniform(-65,-64,size=neurons)

u = b*v
I = np.zeros((neurons)) +10
t = 0
v1 = v
u1 = u
for j in tqdm(range(time.size)):

    data[j] = np.array([v,u])
    time[j] = t

    v1 = v + iz.potential(v,u,I)*step 
    u1 = u + iz.recovery(v,u)*step

    new = iz.update(v1,u1)

    v, u = new[0], new[1]

    t = t + step

""" Store the trajecs difference data with the ith element of coupling force k """
data1 = np.transpose(data,(1,2,0))

""" Get the peak indexes, times and the periods between them """
inds, times, pers = unwrap.get_peaks_indexes(data[:,0,:], thresh, step)
""" Get the phases """
phases = unwrap.unwrap_static_2(total, inds, step,model='Izhikevic')

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
#color = ['b']               # for 1 neuron


""" Plot the phases """
ngplot.phases(phases,color, step)


""" Plot the phases """

#ngplot.phases_diff(0, phases, step, colors=['b'], set_limit=True)  # For neurons = 2
ngplot.phases_diff_3D(0, phases, step)  # For neurons = 100, for neurons = 1, comment both lines



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

