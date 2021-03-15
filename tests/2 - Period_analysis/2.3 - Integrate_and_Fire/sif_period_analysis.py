

""" 
Period Anlysis of Integrate-and-Fire Neuron
===========================================


**Author**: Guilherme M. Toso
**Title**: sif_period_analysis.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization


Description:

    This method calculates Period and Amount of Intervals Means and Standard Deviations of every  
    Stochastic Integrate-and-Fire Neuron of 5000 neurons. Then, the algorithm calculates the grand mean
    and the grand standard deviation of the 5000 neurons.

"""



import sys
import os
import platform
path = os.getcwd()
sys.path.insert(0,path)
from nsc import IntegrateAndFire, ngplot, get_spikes_periods
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from tqdm import tqdm
""" Integrate and Fire Parameters """
vrest = 0.0
r = 1.0
tau = 10.0
threshold = 1.0
I = 2.5

""" Instantiates the Integrate and Fire Model Class """
IF = IntegrateAndFire(vrest,r,tau,threshold)

""" Neurons amount """
neurons = 5000
""" Time Properties """
total = 200

data = np.zeros((total+1,neurons))

u = np.random.uniform(0,0.5,size=neurons)

sigma=0.2

time = np.arange(total)

for j in tqdm(range(time.size)):

    data[j] = u

    next_u = u + IF.lif(u,I) + sigma*u*np.random.normal(0,0.2,size=neurons)

    u = IF.reset(data[j],next_u)

matrix = get_spikes_periods(data.T,threshold,1)

periods = matrix[1]

ngplot.periods_mean(periods)

