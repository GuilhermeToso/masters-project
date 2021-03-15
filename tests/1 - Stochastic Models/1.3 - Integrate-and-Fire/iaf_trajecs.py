""" 
Stochastic Integrate-and-Fire Neuron
================================


**Author**: Guilherme M. Toso
**Tittle**: iaf_trajecs.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization

**Description**:

    Analysis of the Stochastic Integrate-and-Fire Neuron Model

"""


import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import IntegrateAndFire
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

sigma = 0.2

time = np.linspace(0,total,(total+1))

""" Data array """
data = np.zeros((total+1,neurons))

u = np.random.uniform(0,0.5,size=neurons)

for j in tqdm(range(time.size)):

    data[j] = u

    #couple.data = u

    next_u = u + IF.lif(u,I) + sigma*u*np.random.normal(0,0.2,size=neurons) #- couple.synapse(force[i])

    u = IF.reset(data[j],next_u)

""" Store the trajecs difference data with the ith element of coupling force k """
#diff_data[i] = np.abs(data[:,0] - data[:,1])
data1 = np.transpose(data,(1,0))
""" Get the peak indexes, times and the periods between them """
inds, times, pers = unwrap.get_peaks_indexes(data[:,:], threshold, 1)
""" Get the phases """
phases = unwrap.unwrap_static_2(total+1, inds, 1,model='IAF')


""" For neurons = 2 or 100, if is 1, comment this piece of code """
neurons_array = []
for i in range(len(times)):
    neurons_array.append(np.zeros(times[i].size)+i)
ngplot.neural_activity(times, neurons_array,total)


""" Colors to plot the phases """
""" Plasma colormap """
plasma = cm.get_cmap('plasma',neurons)
color = list(plasma.colors) # for 100 neurons
#color = ['b','r']           # for 2 neurons 
#color = ['b']               # for 1 neuron

""" Plot the phases """
ngplot.phases(phases,color, 1)

""" Plot the phases diff """
#ngplot.phases_diff(0,phases,dt,colors=['b']) # For neurons = 2
ngplot.phases_diff_3D(0, phases, 1)         # For neurons = 100, for neurons = 1, comment both lines

ngplot.trajectories(data,time,colors=color)

""" Plot the trajectories difference if neurons = 2 or 100 """
for i in range(neurons-1):
    i = i + 1
    plt.plot(time, np.abs(data1[0] - data1[i]), c = color[i])
plt.xlabel('t [ms]',fontsize=34, labelpad=10)
plt.ylabel(r"$|V_{i} - V_{j}|$ [mV]",fontsize=34, labelpad=30)
plt.yticks(fontsize=34)
plt.xticks(fontsize=34)
plt.grid(True)
plt.show()
