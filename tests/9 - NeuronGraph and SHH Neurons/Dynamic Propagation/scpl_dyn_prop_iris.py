""" 
Neuron Graph - Iris DataSet
=================================

Analysis of the Neuron Graph Propagation and Synchronized Neurons
----------------------------------------------------------------

**Author**: Guilherme M. Toso
**Tittle**: scpl_dyn_prop_iris.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization


**Description**:

    This script uses the Hodgkin-Huxley Biological Neuron Model with Stochastic terms
    to represent the classified data of the Iris Dataset using the Simplicial Complex
    Propagating Labels in a Dynamic Method.
    It uses 5 initial classified data of each class (Iris-Setosa, Versicolor and Virginica),
    then it classifies and measures the accuracy.

"""
""" Dependencies """
import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import HodgkinHuxley, Chemical, SDE, Couple
from nsc import unwrap
from nsc import ngplot
from nsc import SCPL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from tqdm import tqdm
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix
import random
import matplotlib.cm as cm


random.seed(0)
np.random.seed(0)

""" Import the Data """
iris = datasets.load_iris(as_frame=True).frame

""" Instantiate SCPL and preprocess the data """
scpl = SCPL(data=iris, target='target',similarity='Euclidean')
scpl.data_process(set_null=True, label_size=5)


""" Define the amount of neurons """
neurons = scpl.data.shape[0]

""" Set the Hodgkin-Huxley Stochastic Model parameters """
v_na = np.zeros(neurons) + 115
v_k = np.zeros(neurons) - 12
v_l = np.zeros(neurons) + 86
g_na = np.zeros(neurons) + 120
g_k = np.zeros(neurons) + 36
g_l = np.zeros(neurons) + .3
C = np.ones(neurons)
sigma_x = 1.0
sigma_v = 0.5
v_rev = -70
J_q = 0.1
I = 20

""" time properties"""
step = 0.02
t = 0
t_final = 300.0
iterations = int((t_final)/step)

""" Instantiate Hodgkin-Huxley Model Class """
hh = HodgkinHuxley(v_na,v_k,v_l,g_na,g_k,g_l,C)
sde = SDE(step,sigma_v,sigma_x,v_na,v_k,v_l,g_na,g_k,g_l,C)
ch = Chemical(J_q,v_rev)

""" Initialize V,M,N,H,Y """
v = np.random.uniform(0,4,(neurons))
m = hh.m0(v)[2]
n = hh.n0(v)[2]
h = hh.h0(v)[2]
y = ch.y0(v)[2]

""" Define the array where will be stored all variables (V,M,N,H and Y) """
data = np.zeros((iterations,5,neurons))
phases = np.zeros((iterations,neurons))
init = np.array([v,m,n,h,y])
phase_threshold = 40

""" Define the array where will be stored time """
time = np.arange((iterations))
""" Define the number of clusters """
clusters = len(scpl.labels)

""" Determine the coupling force """
k = 0.8

""" Determine the initial connection matrix """
connections = np.zeros(shape=(neurons,neurons))
for i in range(len(scpl.labels)):
    row,col = np.meshgrid(scpl.labels_indexes[scpl.labels[i]], scpl.labels_indexes[scpl.labels[i]].T)
    row = row.flatten()
    col = col.flatten()
    connections[row,col] = k
print(connections.max())
""" Determine the hypersphere size """
diameter = 10*np.std(scpl.numerical,axis=0).max()

""" List the initial labeled data """
list_labeled = []
for i in range(len(scpl.labels)):
    list_labeled += list(scpl.labels_indexes[i])
list_labeled.sort()

arrival = 0
unlabeled_ind = 0

couple = Couple()
for i in range(iterations):

    minimum = [0,0,10**30,0]

    if i == arrival:
        j = 0
        jump = []
        for j in range(len(list_labeled)):
            if unlabeled_ind == list_labeled[j]:
                unlabeled_ind += 1
            else:
                del list_labeled[:1]
                break
        if unlabeled_ind < scpl.data.shape[0]:
            for j in range(len(scpl.labels)):
                distance= scpl.calculate_similarity(
                    scpl.numerical[unlabeled_ind,None,:],
                    scpl.numerical[scpl.labels_indexes[j],:],
                    axis=1
                )

                rows = np.where(distance <= diameter)
                if rows[0].size > 0:
                    min_val = distance[rows].min()
                    if min_val < minimum[2]:
                        minimum[0], minimum[1], minimum[2], minimum[3] = unlabeled_ind, scpl.labels_indexes[j][np.where(distance==min_val)[0][0]], min_val, j
            scpl.labels_indexes[minimum[3]] = np.append(scpl.labels_indexes[minimum[3]],unlabeled_ind)

            for j in range(len(scpl.labels)):
                a,b = np.meshgrid(scpl.labels_indexes[j],scpl.labels_indexes[j])
                connections[a,b] = k
            unlabeled_ind = unlabeled_ind + 1
        arrival = arrival + 30

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



data1 = np.transpose(data,(1,2,0))
viridis = cm.get_cmap('viridis', neurons).colors

indexes_1 = []
for i in range(len(scpl.labels_indexes)):
    indexes_1 += list(scpl.labels_indexes[i])
data2 = np.array([data1[0][i] for i in indexes_1])
ids, times, pers = unwrap.get_peaks_indexes(data1[0,:,:].T, 40, step)
phases = unwrap.unwrap_static_2(data2.shape[1], ids, step,model='HH')
neurons_array = []
for i in range(len(times)):
    neurons_array.append(np.zeros(times[i].size)+i)

ngplot.neural_activity(times, neurons_array,t_final, labeled=scpl.labels_indexes)
# phases = phases.T

""" Plot phases """
ngplot.phases(phases, viridis, step)

""" Plot Trajectories """
ngplot.trajectories(data2.T, time*step)


counts = [0]
value = 0
for i in range(len(scpl.labels_indexes)-1):
    value += len(scpl.labels_indexes[i]) + 1
    counts.append(value)
print(counts)

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















# """ ========================= PARTE 1 ========================= """

# """ Define the time of stop """
# t_stop = 0
# """ Number of neurons that reached the phase thesehold """
# n = 0
# """ Indexes that reached the phase threshold """
# columns = np.array([])
# """ Total array """
# total = np.arange(0,neurons)
# """ times """
# times = np.zeros((neurons)).astype(int)
# #print("times: ", times)

# """ Number of spikes at past """
# spikes = np.zeros((neurons)).astype(int)

# """ Next Spike """
# t_next = np.random.normal(8.81,0.72)


# Synapse = models.Synapse(J_q,v_rev)
# SDE = models.SDE(step,sigma_v,sigma_x,v_na, v_k, v_l, g_na, g_k, g_l, C)

# """ Begin the iteration process """
# for i in range(iterations - 1):
#     #print("Iteration: ", i)
#     data[i] = init

#     V = init[0]
#     M = init[1]
#     N = init[2]
#     H = init[3]
#     Y = init[4]

#             #print('V: ', V)
#     matrix_V = np.zeros(V.size) + V[:,np.newaxis] 
#     #print("matrix_V: \n", matrix_V)
#     VMatrix = matrix_V.T
#     #print("VMatrix: \n", VMatrix)
#     VDifference = (VMatrix.transpose()-VMatrix)
#     #print("VDifference: \n", VDifference)
#     Eletrical_synapse = adjacency*k*VDifference
#     #print("Synapses: \n", Eletrical_synapse)
#     es_sum = np.sum(Eletrical_synapse,axis=1)/neurons  
#     #print("es_sum: ", es_sum)
#     #sys.exit(0)
#     next_V = V + SDE.MembranePotential(V,M,N,H,I) - Synapse.Chemical(Y,V)*step - es_sum
#     next_M = M + SDE.StochasticSodiumActivate(M,V)
#     next_H = H + SDE.StochasticSodiumDeactivate(H,V)
#     next_N = N + SDE.StochasticPotassiumActivate(N,V)
#     next_Y = Y + SDE.StochasticChemicalTransmitter(Y,V)



#     # variables = em.EulerMaruayama(J_q,v_rev,step,sigma_v,sigma_x,v_na,v_k,v_l,g_na,g_k,g_l,C).\
#     #     method(v,m,n,h,y,I,k,adjacent,chosen_labels)

#     # init[0] = variables[0]
#     # init[1] = variables[1]
#     # init[2] = variables[2]
#     # init[3] = variables[3]
#     # init[4] = variables[4]

#     init[0] = next_V
#     init[1] = next_M
#     init[2] = next_N
#     init[3] = next_H
#     init[4] = next_Y

#     #print("V: ", v)
#     t = t + step

#     values = find_indexes.find_first_phase(times,n,data,total,phase_threshold,columns,i)

#     columns = np.array(values[0]).astype(int)
#     #print("Columns: ", columns)
#     n = int(values[1])
#     #print("N: ", n)
#     total = np.array(values[2],dtype=int)
#     times = values[3]
#     #print("Times: ", times)
#     if columns.size>0:
#         phases[i][columns] = 2*np.pi*spikes[columns] + 2*np.pi*(times[columns])/(int(t_next/step)+1)
#     #print("Phases: ", phases[:i])

#     t_stop = i

#     if n == neurons:
#         break

# """ ========================= PARTE 2 ========================= """

# diameter = 10*np.std(scpl.numerical,axis=0).max()
# list_labeled = []
# for i in range(len(scpl.labels)):
#     list_labeled += scpl.labels_indexes[i]
# list_labeled.sort()
# #sys.exit(0)
# d = 0
# t_k = np.zeros((neurons)).astype(int) + times
# #print(t_k)

# t_k_1 = np.zeros((neurons)).astype(int) + int(t_next/step) + 1
# #print(t_k_1)
# e = 0
# f = 0
# g = 1
# for i in tqdm(range(iterations - t_stop)):

#     s = i + t_stop
#     #print("I:", i)

#     minimum = [0,0,10**30,0]
    

#     if i == f:
        
#         j = 0
#         jump = []
#         for j in range(len(list_labeled)):

#             if e == list_labeled[j]: 
#                 #print(e+d)

#                 e = e+1
#             else:
#                 del list_labeled[:j]
#                 break


#                 #print(e+d)
#             # else:
#             #     del list_labeled[:1]
#             #     break

#         if e < 150:
#             #print(e)
#             for j in range(len(scpl.labels)):

#                 distance = np.linalg.norm(
#                     scpl.numerical[e,None,:] - scpl.numerical[scpl.labels_indexes[j],:],
#                 axis=1)
#             # print(distance.shape)
#             # print(distance)
                
#                 rows = np.where(distance <= diameter)
#                 if rows[0].size > 0:
#                     min_val = distance[rows].min()
#                     if min_val < minimum[2]:
#                         minimum[0], minimum[1], minimum[2], minimum[3] = e, scpl.labels_indexes[j][np.where(distance==min_val)[0][0]], min_val, j

#             scpl.labels_indexes[minimum[3]].append(e)
#             #print(len(scpl.labels_indexes[0]))
#             #print(minimum)
#             for j in range(len(scpl.labels)):
#                 a,b = np.meshgrid(scpl.labels_indexes[j],scpl.labels_indexes[j])
#                 adjacency[a,b] = 1
#             e = e + 1
#         f = f + 30
#             #print(adjacent[0])
                
#     #if i+d == 150:
#         #print(adjacent[0,:])
#         #sys.exit(0)   
#     data[s] = init
#     #time[s] = t

#     V = init[0]
#     M = init[1]
#     N = init[2]
#     H = init[3]
#     Y = init[4]

#     matrix_V = np.zeros(V.size) + V[:,np.newaxis] 
#     #print("matrix_V: \n", matrix_V)
#     VMatrix = matrix_V.T
#     #print("VMatrix: \n", VMatrix)
#     VDifference = (VMatrix.transpose()-VMatrix)
#     #print("VDifference: \n", VDifference)
#     Eletrical_synapse = adjacency*k*VDifference
#     #print("Synapses: \n", Eletrical_synapse)
#     es_sum = np.sum(Eletrical_synapse,axis=1)/neurons  
#     #print("es_sum: ", es_sum)
#     #sys.exit(0)
#     next_V = V + SDE.MembranePotential(V,M,N,H,I) - Synapse.Chemical(Y,V)*step - es_sum
#     next_M = M + SDE.StochasticSodiumActivate(M,V)
#     next_H = H + SDE.StochasticSodiumDeactivate(H,V)
#     next_N = N + SDE.StochasticPotassiumActivate(N,V)
#     next_Y = Y + SDE.StochasticChemicalTransmitter(Y,V)


#     # variables = em.EulerMaruayama(J_q,v_rev,step,sigma_v,sigma_x,v_na,v_k,v_l,g_na,g_k,g_l,C).\
#     #     method(v,m,n,h,y,I,k,adjacent,chosen_labels)

#     init[0] = next_V
#     init[1] = next_M
#     init[2] = next_N
#     init[3] = next_H
#     init[4] = next_Y


#     # init[0] = variables[0]
#     # init[1] = variables[1]
#     # init[2] = variables[2]
#     # init[3] = variables[3]
#     # init[4] = variables[4]  

#     t = t+step
#     #print(init[0].shape)
#     #print(data[s][0].shape)
#     indexes = np.where((init[0]>=phase_threshold)&(data[s][0]<phase_threshold))[0]
#     if indexes.size > 0:
        
#         t_k_1[indexes] = s + 1
#         for ind in indexes:
#             phases[time[t_k[ind]+1:t_k_1[ind]],ind] = 2*np.pi*spikes[ind] + 2*np.pi*(time[t_k[ind]+1:t_k_1[ind]] - t_k[ind])/(t_k_1[ind]-t_k[ind])
#             spikes[ind] = spikes[ind] + 1
#             t_k[ind] = s+1
#             t_k_1[ind] = int(np.random.normal(8.81,0.72)/step)+1
#             phases[s+1][ind] = 2*np.pi*spikes[ind] + 2*np.pi*(s+2 - t_k[ind])/(t_k_1[ind])
#         total_indexes = np.arange(0,neurons)
#         cols =  np.delete(total_indexes,indexes)
#         if cols[0].size > 0:
#             phases[s+1][cols] = 2*np.pi*spikes[cols] + 2*np.pi*(s+2 - t_k[cols])/(t_k_1[cols])
    
#     else:

#         phases[s] = 2*np.pi*spikes + 2*np.pi*(s+1 - t_k)/(t_k_1)
#     #print(adjacent[minimum[0],minimum[1]])
# T = step
# print(list_labeled)
# print(scpl.labels_indexes)
# print(len(scpl.labels_indexes[0]))
# print(len(scpl.labels_indexes[1]))
# print(len(scpl.labels_indexes[2]))
# #sys.exit(0)
# data1 = np.transpose(data,(1,2,0))
# viridis = cm.get_cmap('viridis', neurons).colors

# indexes_1 = []
# for i in range(len(scpl.labels_indexes)):
#     indexes_1 += scpl.labels_indexes[i]
# data2 = np.array([data1[0][i] for i in indexes_1])
# ids, times, pers = unwrap.get_peaks_indexes(data1[0,:,:].T, 40, T)
# phases = unwrap.unwrap_static_2(data2.shape[1], ids, T)
# neurons_array = []
# for i in range(len(times)):
#     neurons_array.append(np.zeros(times[i].size)+i)

# ngplot.neural_activity(times, neurons_array,t_final, labeled=scpl.labels_indexes)
# # phases = phases.T

# """ Plot phases """
# ngplot.phases(phases, viridis, step)

# """ Plot Trajectories """
# ngplot.trajectories(data2, time*step)


# counts = [0]
# value = 0
# for i in range(len(scpl.labels_indexes)-1):
#     value += len(scpl.labels_indexes[i]) + 1
#     counts.append(value)
# print(counts)
# # counts = []
# # for i in range(len(scpl.labels_indexes)):
# #     values = sorted()


# """ Get the Phases difference  with group1 as reference"""
# ngplot.phases_diff_3D(counts[0], phases, T)

# """ Get the Phases difference  with group2 as reference"""
# ngplot.phases_diff_3D(counts[1], phases, T)

# """ Get the Phases difference  with group2 as reference"""
# ngplot.phases_diff_3D(counts[2], phases, T)

# """ Get the Trajectories difference  with group1 as reference"""
# ngplot.trajecs_diff_3D(counts[0], data2, T)

# """ Get the Trajectories difference  with group2 as reference"""
# ngplot.trajecs_diff_3D(counts[1], data2, T)

# """ Get the Trajectories difference  with group3 as reference"""
# ngplot.trajecs_diff_3D(counts[2], data2, T)


# # print(adjacent[:0])
# # viridis = cm.get_cmap('viridis', neurons)
# # col = viridis.colors
# # for i in range(neurons):
# #     plt.plot(time, data[:,0,i],color=col[i])
# # plt.show()   


    

# #     plt.plot(time[:t_stop],:t_stop,30])
# #     plt.show()


# # a = dynamic_propagation()

# # style.use('fivethirtyeight')

# #fig = plt.figure()
# #ax1 = fig.add_subplot(1,1,1)

# # def animate(i):


# #     graph_data = open('samplefile.txt','r').read()

# # a = animate(1)
# # x = []
# # y = []


# # fig, ax = plt.subplots()

# # lines = []
# # for i in range(neurons):
# #     for j in range(iterations):
# #         line = ax.plot(time[:j],data[:j,0,i], color = col[i])
# #         lines.append(line)



# # animation = FuncAnimation(fig, lines, frames=np.arange(0,15000,1), interval=20)
# # plt.show()