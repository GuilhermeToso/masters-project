
""" 
Period Anlysis
==============


**Author**: Guilherme M. Toso
**Title**: period_means.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization


Description:

    This method calculates Period and Amount of Intervals Means and Standard Deviations of every  
    Stochastic Hodgkin-Huxley Neuron of 5000 neurons. Then, the algorithm calculates the grand mean
    and the grand standard deviation of the 5000 neurons.

"""

""" Dependencies """

import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import HodgkinHuxley, Chemical, SDE
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from nsc import get_spikes_periods
from nsc import ngplot
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['text.usetex'] = True
from tqdm import tqdm

""" Define the total amount of neurons """
neurons = 5000

""" Define the initial parameters """
v_na = np.zeros(neurons) + 115
v_k = np.zeros(neurons) - 12
v_l = np.zeros(neurons) + 86
g_na = np.zeros(neurons) + 120
g_k = np.zeros(neurons) + 36
g_l = np.zeros(neurons) + .3
c = np.ones(neurons)
sigma = 1
sigma_external = 3.

""" Frequency """    
freq = 50.0

""" Period """
T = 1/freq

""" Define the step value for every step """
step = T

""" Define the total time """
t_max = 300.0

""" Define the number of iterations """
n_iter = int(t_max/step)

""" Instantiate the Hodgkin-Huxley Class """
hh = HodgkinHuxley(v_na, v_k, v_l, g_na, g_k, g_l, c)
sde = SDE(step,sigma_external,sigma, v_na,v_k,v_l,g_na,g_k,g_l,c)

""" Define the initial parameters and the Chemical class """
jch = 0.1
v_reversal = -70
ch = Chemical(jch, v_reversal)

""" Define the initial time t, and the variables V, M, N, H and Y """
t = 0.0
v = np.random.uniform(0.0, 4.0, (neurons))
m = hh.m0(v)[2]
n = hh.n0(v)[2]
h = hh.h0(v)[2]
y = ch.y0(v)[2]

""" Define the array where will be stored all Variables (V, M, N and H) of all Neurons at all times. """
data = np.zeros((n_iter,5,neurons))

""" Initialize the matrix init that contains all Variables of all Neurons at time t """
init = np.array([v,m,n,h,y])

""" Create the array of time """
time = np.zeros((n_iter))



# """ Instantiate the stochastic model """
# sde = models.SDE(step, sigma, sigma_external,v_na, v_k, v_l, g_na, g_k, g_l, C)

""" Begin the Iteration Process """
for i in tqdm(range(len(time))):
    

    """ Stores the matrix init at the data array in the time i """
    data[i] = init
    
    """ The array time at iteration i receives the value of t """
    time[i] = t
    
    """ Define the initial Variables """
    V = init[0]
    M = init[1]
    N = init[2]
    H = init[3]
    Y = init[4]

    """ Set the electrical current I """
    I = 20
    #I = 10

    """ Calculate the next value using Euler-Maruyama Method """
    init[0] = V + sde.membrane_potential(V,M,N,H,I) - ch.synapse(Y,V)*step
    init[1] = M + sde.stochastic_sodium_activate(M,V)
    init[2] = N + sde.stochastic_potassium_activate(N,V)
    init[3] = H + sde.stochastic_sodium_deactivate(H,V)
    init[4] = Y + sde.stochastic_chemical_transmitter(Y,V)

    """ Update Time """
    t = t + step

""" Transpose the data array """
data1 = np.transpose(data,(1,2,0))

""" Calculate the Sodium and Potassium Conductances """
gna1 = 120*(data1[1]**3)*data1[3]
gk1 = 36*(data1[2]**4)

""" Select only the voltages array """
voltages = data1[0][:]#-65

""" Return the indexes and the periods """
matrix = get_spikes_periods(voltages, 40, step)
periods = matrix[1]
ngplot.periods_mean(periods)
