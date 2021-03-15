""" 
Accuracy per Parameter Combinations
===================================

**Author**: Guilherme M. Toso
**Tittle**: par_combination.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization

**Description**:

    This script takes 5 artificial datasets and apply them in the Neuron Synchronization Competition
    algrithm, by varying the parameters combinations of the limit connections amount (Neighbors),
    the speed of the search sphere radius (Expand), and the total amount of initial classified
    samples (Seeds). 

"""



import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import NSC, ngplot, unwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from tqdm import tqdm
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import multiprocessing

def concat_data(x,y):

    x_pd = pd.DataFrame(x,columns=['X','Y'])
    y_pd = pd.DataFrame(y,columns=['target'])

    df = pd.concat([x_pd,y_pd],axis=1)

    return df

""" Define the range of the params """

def multi_get_params(data,name):

    print("Process ID: ", os.getpid())

    seeds = [2,5,10,12,15,20,22,25,30,32,35,40,42,45,50]
    expand = [2,3,4,5,10,20,30,40,50,75,100,200,300,400,500]
    neighbors = [5,10,15,20,25,50]

    epochs = 600

    features = data.drop(columns=['target'],axis=1)

    numeric = features.select_dtypes(include=np.number)

    numeric_names = numeric.columns

    data.loc[:,numeric_names] = (numeric-numeric.mean())/numeric.std()


    # Cria a instância do modelo
    ng = NSC(data=data, target='target', similarity='Euclidean',
    model='Izhikevic', alpha = 0.1, w_step = 0.3, time_step=0.5, print_info=False,
    print_steps=False, beta=2.0, gamma=1.5)

    labels = np.unique(ng.data['target'])

    accuracy = np.zeros(shape=(len(neighbors), len(labels), len(seeds),len(expand)))
    accuracy_mean = np.zeros(shape=(50,len(labels)))
    
    # Para cada valor de vizinhos
    for n in tqdm(range(len(neighbors))):
        print("{} data neighbors: {}/6".format(name,n))
        ng.neighbors = neighbors[n]
        # Para cada valor de seeds
        for s in tqdm(range(len(seeds))):
            print("{} data seeds: {}/15".format(name,s))
            # Para cada valor de expand
            for e in tqdm(range(len(expand))):
                print("{} data expand: {}/15".format(name,e))
                ng.search_expand = expand[e]
                
                # Para 50 iterações
                for i in range(50):
                    
                    # Preprocessa os dados
                    ng.preprocess_data(not_null=seeds[s], standarlize=False)
                    
                    # Fit
                    ng.fit(epochs,ng.numerical)
                    
                    diag =  np.diagonal(ng.confusion_matrix)
                    if diag.size==len(ng.labels_list):
                        accuracy_mean[i,:] = diag
                    elif diag.size > len(ng.labels_list):
                        accuracy_mean[i,:] = diag[-len(ng.labels_list):]
                    
                    ng.data = data
                    
                    ng.y_predicted = -np.ones(shape=ng.neurons)

                    ng.degree_out = np.zeros(shape=(ng.neurons,len(ng.labels_list)+1))
                    ng.labels_array = -np.ones(shape=(ng.neurons,ng.neurons))
                    ng.incident_degree = np.zeros(shape=(ng.neurons,len(ng.labels_list)))
                    ng.inner_degree = {}
                    ng.disputed = np.array([])
                    ng.capacity = np.zeros(shape=(ng.neurons))

                accuracy[n,:,s,e] = np.mean(accuracy_mean,axis=0)

    np.save(name,accuracy)

if __name__ == "__main__":
    

    x_clusters_09,y_clusters_09 = datasets.make_blobs(n_samples=400,centers=3,n_features=2, cluster_std=0.9)
    clusters_09 = concat_data(x_clusters_09,y_clusters_09)
    
    x_clusters_12,y_clusters_12 = datasets.make_blobs(n_samples=400,centers=3,n_features=2, cluster_std=1.2)
    clusters_12 = concat_data(x_clusters_12,y_clusters_12)
    
    x_circles,y_circles = datasets.make_circles(n_samples=400,noise=0.05,factor=0.5)
    circles = concat_data(x_circles, y_circles)

    x_moons,y_moons = datasets.make_moons(n_samples=400,noise=0.1)
    moons = concat_data(x_moons, y_moons)

    x_classification, y_classification = datasets.make_classification(n_samples=400,n_features=2,n_informative=2,n_redundant=0,n_clusters_per_class=1)
    classification = concat_data(x_classification,y_classification)

    dic_of_data = {'Cluster_09':clusters_09, 'Cluster_12':clusters_12, 'Circles':circles, 'Moons':moons, 'Classification':classification}

    keys = list(dic_of_data.keys())
    print(keys)
    datas = list(dic_of_data.values())

    processes = []

    for i in tqdm(range(5)):

        p = multiprocessing.Process(target=multi_get_params,args=(datas[i],keys[i],))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()