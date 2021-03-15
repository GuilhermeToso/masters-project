
""" 
Period Anlysis of the Hindmarsh-Rose Neuron
===========================================


**Author**: Guilherme M. Toso
**Title**: shr_period_analysis.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization


Description:

    This method calculates Period and Amount of Intervals Means and Standard Deviations of every  
    Stochastic Hindmarsh-Rose Neuron of 5000 neurons. Then, the algorithm calculates the grand mean
    and the grand standard deviation of the 5000 neurons.

"""

""" Dependencies """

import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import HindmarshRose, ngplot, get_spikes_periods
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['text.usetex'] = True
from tqdm import tqdm

""" Create the model """
hr = HindmarshRose(1,3.0,1.0,5.0,0.001,1.6,-1.6,1)

""" Create the neurons """
neurons = 5000

""" Time properties """
t,t_f,dt = 0.0,50,0.01
n_iter = int(t_f/dt)

""" Initialize x, y and z """
init = np.random.uniform(-0.5,.5,(3,neurons))

""" Define the array where will be stored all Variables (X,Y and Z) of all Neurons at all times. """
data = np.zeros((n_iter,3,neurons))


""" Create the array of time """
time = np.zeros((n_iter))

sigma = 1

for i in tqdm(range(n_iter)):
    
    data[i]=init
    time[i] = t

    X, Y, Z = init[0], init[1], init[2]

    next_X = X + hr.potential(X,Y,Z)*dt + sigma*X*np.random.uniform(0,dt,size=X.size)
    next_Y = Y + hr.fast_ion_channel(X,Y)*dt + sigma*X*np.random.uniform(0,dt,size=X.size)
    next_Z = Z + hr.slow_ion_channel(X,Z)*dt + sigma*X*np.random.uniform(0,dt,size=X.size)
    
    init[0,:] = next_X
    init[1,:] = next_Y
    init[2,:] = next_Z

    t = t + dt

voltage = data[:,0,:]

matrix = get_spikes_periods(voltage.T,1,dt)

periods = matrix[1]

ngplot.periods_mean(periods)