import sys
import os
import platform

system = platform.system()

current_dir = os.getcwd()
print(current_dir)

if system == 'Windows':
    path_dir = current_dir.split("\\Neurons")[0] + "\\Neurons"
else:
    path_dir = a.split("/Neurons")[0] + "/Neurons"

sys.path.append(path_dir)
from model import NGraph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from model import ngplot
from model import unwrap
from tqdm import tqdm
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

def concat_data(x,y):

    x_pd = pd.DataFrame(x,columns=['X','Y'])
    y_pd = pd.DataFrame(y,columns=['target'])

    df = pd.concat([x_pd,y_pd],axis=1)

    return df

def y_true(y):
    
    uniques = np.unique(y)
    amount = np.array([])
    for i in range(uniques.size):
        amount = np.append(amount,np.where(y==uniques[i])[0].size)
    
    return amount.astype(int)


""" Create Artificial Datasets """

x_clusters,y_clusters = datasets.make_blobs(n_samples=500,centers=3,n_features=2, cluster_std=1.2)
clusters = concat_data(x_clusters, y_clusters)


x_circles,y_circles = datasets.make_circles(n_samples=500,noise=0.05,factor=0.5)
circles = concat_data(x_circles,y_circles)

x_moons,y_moons = datasets.make_moons(n_samples=500,noise=0.1)
moons = concat_data(x_moons, y_moons)

x_classification, y_classification = datasets.make_classification(n_samples=500,n_features=2,n_informative=2,n_redundant=0,n_clusters_per_class=1)
classification = concat_data(x_classification,y_classification)

x_gaussian, y_gaussian = datasets.make_gaussian_quantiles(n_samples=500,n_features=2)
gaussian = concat_data(x_gaussian,y_gaussian)


print("Cluster Real Classified: ", y_true(y_clusters))
print("Circles Real Classified: ", y_true(y_circles))
print("Moons Real Classified: ", y_true(y_moons))
print("Classification Real Classified: ", y_true(y_classification))
print("Gaussian Real Classified: ", y_true(y_gaussian))


dic_of_data = {
    'cluster_seeds':clusters,
    'circles_seeds':circles,
    'moons_seeds':moons,
    'classification_seeds':classification,
    'gaussian_seeds':gaussian
}

beta = 2
gamma = 2

seeds = np.linspace(1,10,num=10)*0.01
num = seeds*500
num = num.astype(int)

epochs = 700


for key,data in dic_of_data.items():

    ng = NGraph(data=data, target='target', similarity='Euclidean',
    model='Izhikevic', alpha=0.1, w_step=3, time_step=0.5, print_info=False,
    print_steps=False, beta=beta, gamma=gamma)

    variables = ng.neuron.variables

    labels = np.unique(ng.data['target'])
    
    ng.search_expand = 200
    ng.threshold = 6*ng.data.std().max()/ng.search_expand
    
    accuracy = np.zeros(shape=(num.size, len(labels)))
    accuracy_mean = np.zeros(shape=(50,len(labels)))
    accuracy_std = np.zeros(shape=(num.size,len(labels)))
    
    for i in tqdm(range(num.size)):
        
        
        for j in tqdm(range(50)):
            
            ng.preprocess_data(not_null=num[i])

            ng.fit(epochs, ng.numerical)

            diag = np.diagonal(ng.confusion_matrix)

            if diag.size == len(ng.labels_list):
                accuracy_mean[j,:] = diag
            elif diag.size > len(ng.labels_list):
                accuracy_mean[j,:] = diag[-len(ng.labels_list):]
            
            ng.data = data
            ng.neuron.variables = variables
            ng.y_predicted = -np.ones(shape=ng.neurons)
            ng.degree_out = np.zeros(shape=(ng.neurons, len(ng.labels_list)+1))
            ng.labels_array = -np.ones(shape=(ng.neurons,ng.neurons))
            ng.incident_degree = np.zeros(shape=(ng.neurons, len(ng.labels_list)))
            ng.inner_degree = {}
            ng.graph.vertexes = np.array([])
            ng.disputed = np.array([])
        accuracy[i,:] = np.mean(accuracy_mean,axis=0)
        accuracy_std[i,:] = np.std(accuracy_mean,axis=0)
        
    np.save(key,accuracy)
    np.save(key+'_std', accuracy_std)

print("Cluster Real Classified: ", y_true(y_clusters))
print("Circles Real Classified: ", y_true(y_circles))
print("Moons Real Classified: ", y_true(y_moons))
print("Classification Real Classified: ", y_true(y_classification))
print("Gaussian Real Classified: ", y_true(y_gaussian))