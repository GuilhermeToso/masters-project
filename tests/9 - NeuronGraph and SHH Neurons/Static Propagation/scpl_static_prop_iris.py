""" 
Neuron Graph - Iris DataSet
=================================

Analysis of the Neuron Graph Propagation and Synchronized Neurons
----------------------------------------------------------------

**Author**: Guilherme M. Toso
**Tittle**: static_propagation_iris.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization


**Description**:

    This script uses the Hodgkin-Huxley Biological Neuron Model with Stochastic terms
    to represent the classified data of the Iris Dataset using the Neuron Graph Method.
    It uses 5 initial classified data of each class (Iris-Setosa, Versicolor and Virginica),
    then it classifies and measures the accuracy.

"""
""" Dependencies """
import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import HodgkinHuxley,Chemical,SDE,Couple
from nsc import unwrap
from nsc import ngplot
from nsc import SCPL
import numpy as np
import cupy as cp
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix
import random
import matplotlib.cm as cm

random.seed(0)
np.random.seed(0)

""" Import the Data """
iris = datasets.load_iris(as_frame=True).frame

""" Instantiate SCPL, preprocess and fit the data """
scpl = SCPL(data=iris, target='target',similarity='Euclidean')
scpl.data_process(set_null=True, label_size=7)
scpl.fit(scpl.numerical,100)

""" Define the neurons """
neurons = scpl.data.shape[0]

""" Set the Hodgkin-Huxley Stochastic Model parameters """
v_na = np.zeros(neurons) + 115
v_k = np.zeros(neurons) - 12
v_l = np.zeros(neurons) + 86
g_na = np.zeros(neurons) + 120
g_k = np.zeros(neurons) + 36
g_l = np.zeros(neurons) + .3
C = np.ones(neurons)

""" Frequency """    
freq = 50.0
# OKAY

""" Period """
T = 1/freq
# OKAY

""" Define the step value for every step """
step = T
# OKAY

""" Define the total time """
t_max = 300.0
# OKAY

""" Define the number of iterations """
n_iter = int(t_max/step)
# OKAY

sigma = 1.
sigma_external = 0.5
# OKAY

""" Instantiate the Hodgkin-Huxley Class """
hh = HodgkinHuxley(v_na, v_k, v_l, g_na, g_k, g_l, C)
sde = SDE(step,sigma_external,sigma,v_na, v_k, v_l, g_na, g_k, g_l, C)
jch = 0.1
v_rev = -70
ch = Chemical(jch,v_rev)
# OKAY

""" Define the initial time t, and the variables V, M, N, and H """
t = 0.0
v = np.random.uniform(0.0, 4.0, (neurons))
m = hh.m0(v)[2]
n = hh.n0(v)[2]
h = hh.h0(v)[2]
y = ch.y0(v)[2]
# OKAY
""" Define the array where will be stored all Variables (V, M, N and H) of all Neurons at all times. """
data = np.zeros((n_iter,5,neurons))
# OKAY

""" Initialize the matrix init that contains all Variables of all Neurons at time t """
init = np.array([v,m,n,h,y])
# OKAY

""" Create the array of time """
time = np.zeros((n_iter))
# OKAY

k = 0.8
connections = np.zeros(shape=(neurons,neurons))
for i in range(len(scpl.labels)):
    scpl.labels_indexes[scpl.labels[i]] = np.append(scpl.labels_indexes[scpl.labels[i]],scpl.classified[scpl.labels[i]])
    row, col = np.meshgrid(scpl.labels_indexes[scpl.labels[i]], scpl.labels_indexes[scpl.labels[i]].T)
    row = row.flatten()
    col = col.flatten()
    connections[row,col] = k
couple = Couple()

""" Begin the Iteration Process """
for i in tqdm(range(len(time))):
    
    """ Stores the matrix init at the data array in the time i """
    data[i] = init
    
    """ The array time at iteration i receives the value of t """
    time[i] = t
    
    """ Define the initial Variables """
    v = init[0]
    m = init[1]
    n = init[2]
    h = init[3]
    y = init[4]
    
    couple.data = v

    """ Set the electrical current I """
    current = 20
    
    next_v = v + sde.membrane_potential(v,m,n,h,current) - ch.synapse(y,v)*step - couple.synapse(connections)
    next_m = m + sde.stochastic_sodium_activate(m,v)
    next_h = h + sde.stochastic_sodium_deactivate(h,v)
    next_n = n + sde.stochastic_potassium_activate(n,v)
    next_y = y + sde.stochastic_chemical_transmitter(y,v)

    init[0] = next_v
    init[1] = next_m
    init[2] = next_n
    init[3] = next_h
    init[4] = next_y

    
    """ Update Time """
    t = t + step

""" Transpose the data array """
data1 = np.transpose(data,(1,2,0))


""" Calculate the Sodium and Potassium Conductances """
gna1 = 120*(data1[1]**3)*data1[3]
gk1 = 36*(data1[2]**4)

viridis = cm.get_cmap('viridis', neurons).colors

""" Organize the data """
indexes = []
for i in range(len(scpl.labels_indexes)):
    indexes += list(scpl.labels_indexes[i])

data2 = np.array([data1[0][i] for i in indexes])

""" Get the peak indexes, times and the periods between them """
inds, times, pers = unwrap.get_peaks_indexes(data1[0,:,:].T, 40, T)

neurons_array = []
for i in range(len(times)):
    neurons_array.append(np.zeros(times[i].size)+i)

ngplot.neural_activity(times, neurons_array,t_max, labeled=scpl.labels_indexes)

""" Get the phases """
phases = unwrap.unwrap_static_2(data2.shape[1], inds, T,model='HH')

""" Plot phases """
ngplot.phases(phases, viridis, T)

""" Plot Trajectories """
ngplot.trajectories(data2.T, time)


counts = [0]
value = 0
for i in range(len(scpl.labels_indexes)-1):
    value += len(scpl.labels_indexes[i]) + 1
    counts.append(value)




""" Get the Phases difference  with group1 as reference"""
ngplot.phases_diff_3D(counts[0], phases, T)

""" Get the Phases difference  with group2 as reference"""
ngplot.phases_diff_3D(counts[1], phases, T)

""" Get the Phases difference  with group2 as reference"""
ngplot.phases_diff_3D(counts[2], phases, T)

""" Get the Trajectories difference  with group1 as reference"""
ngplot.trajecs_diff_3D(counts[0], data2, T)

""" Get the Trajectories difference  with group2 as reference"""
ngplot.trajecs_diff_3D(counts[1], data2, T)

""" Get the Trajectories difference  with group3 as reference"""
ngplot.trajecs_diff_3D(counts[2], data2, T)


