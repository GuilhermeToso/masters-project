
""" 
Period Anlysis of the Izhikevich Neuron
===========================================


**Author**: Guilherme M. Toso
**Title**: izhikevich_period_analysis.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization


Description:

    This method calculates Period and Amount of Intervals Means and Standard Deviations of every  
    Chaotic Izhikevich Neuron of 5000 neurons. Then, the algorithm calculates the grand mean
    and the grand standard deviation of the 5000 neurons.

"""

import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import Izhikevic, ngplot, get_spikes_periods
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from tqdm import tqdm

""" Neurons Amount """
neurons = 5000

""" Izhikevic Parameters """
a = 0.02
b = 0.2
c = -55
d = 0.95

""" Instantiate Izhikevic Model """
iz = Izhikevic(a,b,c,d)

""" Time Properties """
t = 0
amount = 400
total = amount + 1
time = np.zeros((total))
step = 1

time = np.zeros(shape=(total))

""" Data Properties """
data = np.zeros((total,2,neurons))

""" Define os x, y e I iniciais """
v = np.random.uniform(-65,-64,size=neurons)

u = b*v
I = np.zeros((neurons)) + 10
t = 0

for j in tqdm(range(time.size)):

    data[j] = np.array([v,u])
    time[j] = t

    v1 = v + iz.potential(v,u,I)*step
    u1 = u + iz.recovery(v,u)*step

    new = iz.update(v1,u1)

    v, u = new[0], new[1]

    t = t + step


matrix = get_spikes_periods(data[:,0,:].T,-55,step)

periods = matrix[1]

ngplot.periods_mean(periods)