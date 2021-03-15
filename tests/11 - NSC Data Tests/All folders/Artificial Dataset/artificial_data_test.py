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
from model import NGraph, PCC
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
#np.random.seed(0)
import multiprocessing

def concat_data(x,y):

    x_pd = pd.DataFrame(x,columns=['X','Y'])
    y_pd = pd.DataFrame(y,columns=['target'])

    df = pd.concat([x_pd,y_pd],axis=1)

    return df

""" Define the range of the params """

def multi_get_params(data,name):

    print("Process ID: ", os.getpid())

    beta = np.linspace(0.1,2,num=10)
    gamma = np.linspace(0.1,2,num=10)
    expand = [50, 100, 200, 300]

    epochs = 600

    # Cria a instância do modelo
    ng = NGraph(data=data, target='target', similarity='Euclidean',
    model='Izhikevic', alpha = 0.1, w_step = 0.3, time_step=0.5, print_info=False,
    print_steps=False)

    labels = np.unique(ng.data['target'])

    accuracy = np.zeros(shape=(len(expand),len(labels),beta.size,gamma.size))
    accuracy_mean = np.zeros(shape=(50,len(labels)))
    
    # Para cada valor de expand
    for e in range(len(expand)):
        print("{} data expand: {}/4".format(name,e))
        ng.search_expand = expand[e]
        threshold = 6*ng.data.std().max()/ng.search_expand
        # Para cada valor de beta
        for b in range(beta.size):
            print("{} data beta: {}/10".format(name,b))
            ng.beta = beta[b]
            # Para cada valor de gamma
            for g in range(gamma.size):
                print("{} data gamma: {}/10".format(name,g))
                ng.gamma = gamma[g]
                # Para 100 iterações
                for i in range(50):
                    ng.threshold = threshold
                    # Preprocessa os dados
                    ng.preprocess_data(not_null=25)
                    # Fit
                    ng.fit(epochs,ng.numerical)
                    diag =  np.diagonal(ng.confusion_matrix)
                    if diag.size==len(ng.labels_list):
                        accuracy_mean[i,:] = diag
                    elif diag.size > len(ng.labels_list):
                        accuracy_mean[i,:] = diag[-len(ng.labels_list):]
                    #print("Acc: ", diag)
                    ng.data = data
                    #ng.neuron.variables = variables
                    #print("Vars: ", ng.neuron.variables)
                    ng.y_predicted = -np.ones(shape=ng.neurons)



                    ng.degree_out = np.zeros(shape=(ng.neurons,len(ng.labels_list)+1))
                    ng.labels_array = -np.ones(shape=(ng.neurons,ng.neurons))
                    ng.incident_degree = np.zeros(shape=(ng.neurons,len(ng.labels_list)))
                    ng.inner_degree = {}
                    ng.graph.vertexes = np.array([])
                    ng.disputed = np.array([])
                accuracy[e,:,b,g] = np.mean(accuracy_mean,axis=0)

    np.save(name,accuracy)

if __name__ == "__main__":
    
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

    dic_of_data = {'Cluster':clusters, 'Circles':circles, 'Moons':moons, 'Classification':classification, "Gaussian":gaussian}

    keys = list(dic_of_data.keys())
    print(keys)
    datas = list(dic_of_data.values())

    processes = []

    for i in tqdm(range(5)):

        p = multiprocessing.Process(target=multi_get_params,args=(datas[i],keys[i],))
        p.start()
        processes.append(p)
        print(p)
    
    for p in processes:
        p.join()