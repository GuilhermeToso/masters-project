import sys
import os
import platform

system = platform.system()

current_dir = os.getcwd()

if system == 'Windows':
    path_dir = current_dir.split("\\Neurons")[0] + "\\Neurons"
else:
    path_dir = a.split("/Neurons")[0] + "/Neurons"

sys.path.append(path_dir)
from model import NeuronGraph, NGraph
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from model import ngplot
from model import unwrap
from tqdm import tqdm
import pandas as pd
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix

#np.random.seed(0)


def concat_data(x,y):

    x_pd = pd.DataFrame(x,columns=['X','Y'])
    y_pd = pd.DataFrame(y,columns=['target'])

    df = pd.concat([x_pd,y_pd],axis=1)

    return df

x_clusters,y_clusters = datasets.make_blobs(n_samples=400,centers=3,n_features=2, cluster_std=1.2)
clusters = concat_data(x_clusters,y_clusters)

x_circles,y_circles = datasets.make_circles(n_samples=400,noise=0.05,factor=0.5)
circles = concat_data(x_circles, y_circles)

x_moons,y_moons = datasets.make_moons(n_samples=400,noise=0.1)
moons = concat_data(x_moons, y_moons)

x_classification, y_classification = datasets.make_classification(n_samples=400,n_features=2,n_informative=2,n_redundant=0,n_clusters_per_class=1)
classification = concat_data(x_classification,y_classification)

x_gaussian, y_gaussian = datasets.make_gaussian_quantiles(n_samples=400,n_features=2)
gaussian = concat_data(x_gaussian, y_gaussian)




iris = datasets.load_iris(as_frame=True).frame

data = iris

ng = NeuronGraph(
    data = data, target='target', similarity='Euclidean',
    model = 'Izhikevic',alpha = 0.2,
    w_step = 0.6,time_step=0.5, expand=50,print_info=False, print_steps=False,beta=2,gamma=1.5, neighbors=10)
 
ng.preprocess_data(shuffle=False,not_null=12, standarlize=False)

epochs = 700

#ng.fit(epochs,ng.numerical)
#print("Confusion Matrix: \n", ng.confusion_matrix)

ng.initialize_weights(ng.numerical)

epochs = 700
ng.neuron.potential = np.zeros(shape=(epochs,len(ng.neuron.variables),ng.neurons))
i = 0
j = 0
ng.neuron.connections = ng.weights
max_std = 0
for label in ng.labels_list:
    ng.classified_number = ng.classified_number + ng.indexes_of['label'][label].size
    std = ng.data.loc[ng.indexes_of['label'][label],:].std().max()
    if std > max_std:
        max_std = std
    #stds.append(np.std(ng.data[ng.indexes_of['label'][label],:],axis=0).max())
ng.threshold = 6*max_std/ng.search_expand


while i<epochs or ng.classified_number<ng.data.shape[0]:

    ng.classified_number = 0

    if j >= len(ng.indexes_of['unlabel']):
        #print("It has reached the end of the Unlabeled array, returning to index 0 \n")
        j = 0
    if ng.indexes_of['unlabel'].size > 0:
        ng.connection_search(j,ng.numerical)
        print("Classificado: ", ng.classified)
        ng.find_multiple_incidence(j)
    ng.update_weights()

    ng.neuron.variables = ng.neuron.dynamic(ng.time_step, ng.weights, ng.neuron.variables)
    ng.neuron.potential[i] = np.array([var for var in list(ng.neuron.variables)])
    
    if not isinstance(ng.classified,type(None)):
        #before = ng.indexes_of['unlabel']
        ng.indexes_of['unlabel'] = np.setdiff1d(ng.indexes_of['unlabel'],ng.classified)
        #if ng.indexes_of['unlabel'].size == 0 and before.size >0:
         #   print("Time that happen: ", i*0.5)
    #    print("Unlabel Size: ", ng.indexes_of['unlabel'].size)
    else:
        j = j+1
    i = i+1
    if i == epochs:
        break
    if ng.indexes_of['unlabel'].size != 0:
        ng.threshold = ng.threshold + 6*max_std/ng.search_expand
    
    for label in ng.labels_list:
        ng.classified_number = ng.classified_number + ng.indexes_of['label'][label].size
    
    
    print("\nDisputados: ", ng.disputed.size)
    print('Iteration: ', i)
    print('classificados: ', ng.classified_number)
    print('unlabel: \n', ng.indexes_of['unlabel'].size)

print(ng.indexes_of['label'])     
ng.y_predicted = -np.ones(shape=(ng.data.shape[0]))
for label in ng.labels_list:
    ng.y_predicted[ng.indexes_of['label'][label]] = label
    print(ng.indexes_of['label'][label].size)
conf = confusion_matrix(ng.Y, ng.y_predicted)
print("The confusion matrix is: \n", conf)
diag = np.diagonal(conf)
print("Diagonal: ", diag
)
print(ng.capacity)
# for label in ng.labels_list:
#     print(ng.weights[np.where(ng.labels_array==label)])

x = ng.numerical[:10]
print(x)
a = ng.predict(x,ng.numerical)
print(a)

ngplot.trajectories(ng.neuron.potential[:,0,:],np.arange(epochs)*ng.time_step)
ngplot.trajectories(ng.neuron.potential[:,0,ng.indexes_of['label'][0]],np.arange(epochs)*ng.time_step)
ngplot.trajectories(ng.neuron.potential[:,0,ng.indexes_of['label'][1]],np.arange(epochs)*ng.time_step)
ngplot.trajectories(ng.neuron.potential[:,0,ng.indexes_of['label'][2]],np.arange(epochs)*ng.time_step)

""" Get the peak indexes, times and the periods between them """
inds, times, pers = unwrap.get_peaks_indexes(ng.neuron.potential[:,0,:], -40, ng.time_step)
#print(inds)

""" Get the phases """
phases = unwrap.unwrap_static_2(ng.neuron.potential.shape[0], inds, ng.time_step, model='Izhikevic')


neurons_array = []
for i in range(len(times)):
    neurons_array.append(np.zeros(times[i].size)+i)
ngplot.neural_activity(times, neurons_array,epochs*0.5)
print(ng.indexes_of['label'][0])
print(ng.indexes_of['label'][0][0])

print(ng.indexes_of['label'][1])
print(ng.indexes_of['label'][1][0])
print(ng.indexes_of['label'][2])
print(ng.indexes_of['label'][2][0])

""" Get the Phases difference  with group1 as reference"""
ngplot.phases_diff_3D(int(ng.indexes_of['label'][0][0]), phases, ng.time_step)

""" Get the Phases difference  with group2 as reference"""
ngplot.phases_diff_3D(int(ng.indexes_of['label'][1][0]), phases, ng.time_step)

""" Get the Phases difference  with group2 as reference"""
ngplot.phases_diff_3D(int(ng.indexes_of['label'][2][0]), phases, ng.time_step)
