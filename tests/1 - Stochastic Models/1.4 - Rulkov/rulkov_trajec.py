""" 
Chaotic Rulkov Neuron
================================


**Author**: Guilherme M. Toso
**Tittle**: rulkov_trajecs.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization

**Description**:

    Analysis of the Chaotic Rulkov Neuron Model

"""

import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import Rulkov
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

""" Rulkov Parameters """
a = 4.2
alpha = 0.001
sigma = -1.2


""" Instantiate Rulkov Model """
rk = Rulkov(a,alpha,sigma)


""" Time Properties """
step = 1
t = 2000
time = np.linspace(0,t,(t+1))
""" Data Properties """
data = np.zeros((t+1,2,neurons))

""" Define os x, y e I iniciais """
x = np.random.uniform(-1,-1.2,size=neurons)
y = np.zeros((neurons)) - 2.9
I = np.zeros((neurons))

for j in tqdm(range(time.size)):

    
    fast = rk.fx(x,y,I)
    slow = rk.fy(x,y)

    x = fast
    y = slow

    data[j] = np.array([x,y])

""" Store the trajecs difference data with the ith element of coupling force k """
data1 = np.transpose(data,(1,2,0))

""" Get the peak indexes, times and the periods between them """
inds, times, pers = unwrap.get_peaks_indexes(data[:,0,:], -1.7, 1)
""" Get the phases """
phases = unwrap.unwrap_static_2(t+1, inds, 1,model='Rulkov')


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
ngplot.phases(phases,color, 1)

""" Plot the phases diff """
#ngplot.phases_diff(0, phases, 1, colors=['b'], set_limit=True) # For neurons = 2
ngplot.phases_diff_3D(0, phases, 1) # For neurons = 100, for neurons = 1, comment both lines

ngplot.trajectories(data[:,0,:],time,colors=color)

""" Plot the trajectories difference """
for i in range(neurons-1):
    i = i + 1
    plt.plot(time, np.abs(data1[0][0] - data1[0][i]), c = color[i])
plt.xlabel('t [ms]',fontsize=34, labelpad=10)
plt.ylabel(r"$|V_{i} - V_{j}|$ [mV]",fontsize=34, labelpad=30)
plt.yticks(fontsize=34)
plt.xticks(fontsize=34)
plt.grid(True)
plt.show()
