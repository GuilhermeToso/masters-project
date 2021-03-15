
""" 
Period Anlysis of the Courbage-Nekorkin-Vdovin Neuron
===========================================


**Author**: Guilherme M. Toso
**Title**: cnv_period_analysis.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization


Description:

    This method calculates Period and Amount of Intervals Means and Standard Deviations of every  
    Chaotic Courbage-Nekorkin-Vdovin Neuron of 5000 neurons. Then, the algorithm calculates the grand mean
    and the grand standard deviation of the 5000 neurons.

"""


import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import CNV, ngplot, get_spikes_periods
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from tqdm import tqdm

""" Neurons Amount """
neurons = 5000

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

matrix = get_spikes_periods(data[:,0,:].T,0.2,1)

periods = matrix[1]

ngplot.periods_mean(periods)