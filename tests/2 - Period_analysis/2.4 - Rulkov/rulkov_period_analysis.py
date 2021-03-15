

""" 
Period Anlysis of the Rulkov Neuron
===========================================


**Author**: Guilherme M. Toso
**Title**: rulkov_period_analysis.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization


Description:

    This method calculates Period and Amount of Intervals Means and Standard Deviations of every  
    Chaotic Rulkov Neuron of 5000 neurons. Then, the algorithm calculates the grand mean
    and the grand standard deviation of the 5000 neurons.

"""


import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import Rulkov, ngplot, get_spikes_periods
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from tqdm import tqdm

""" Neurons Amount """
neurons = 5000

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

matrix =get_spikes_periods(data[:,0,:].T,-1.7,1)

periods = matrix[1]

ngplot.periods_mean(periods)