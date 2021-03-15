""" 
Simplicial Complex Propagating Labels - Iris DataSet
===================================================

Analysis of the Accuracy per Labels Size comparing with other
Algorithms.
------------------------------------------------------------

**Author**: Guilherme M. Toso
**Tittle**: algs_compare_scpl.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization

**Description**:

    This script uses the Simplicial Complex Propagating Labels Algorithm to classify the unlabeled data
    and varies the label size then analyzes the accuracy and threshold values between (6*std.max()/epochs) 
    and 6*std.max().

"""

""" Dependencies """
import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import unwrap
from nsc import ngplot
from nsc import SCPL
import numpy as np
import cupy as cp
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import random


""" Import the Data Set """
iris = datasets.load_iris(as_frame=True).frame


""" Get X (features) and Y (target) """
X = iris.to_numpy()[:,:-1]
y = -np.ones(shape=X.shape[0])
target_true = iris['target']

""" Get the labels """
labels, counts = np.unique(iris.to_numpy()[:,-1], return_counts=True)
labels = labels.astype(int)

scpl = SCPL(data=iris, target='target',similarity='Euclidean')
seeds = np.arange(20).astype(int) + 1

accuracies_mean = np.zeros(shape=(3,3,seeds.size))
accuracies_std = np.zeros(shape=(3,3,seeds.size))
diameters_mean = np.zeros(shape=(seeds.size))
diameters_std = np.zeros(shape=(seeds.size))


for i in tqdm(range(seeds.size)):

    acc = np.zeros(shape=(3,3,100))
    diam = np.zeros(shape=(100))

    for j in range(100):
        print(j)

        """ Simplicial Complex Propagating Labels """

        scpl.data_process(set_null=True, label_size=seeds[1])
        scpl.fit(scpl.numerical)

        acc[0,:,j] = scpl.accuracy
        diam[j] = scpl.threshold_list[-1]

        scpl.data['target'] = target_true

        """ Label Propagation """
        lp = LabelPropagation()

        indexes = []
        for k in range(len(scpl.labels)):
            label_indexes = np.where(scpl.y_true == scpl.labels[k])[0]
            random_indexes = np.random.choice(label_indexes, size=seeds[i], replace=False)
            y[random_indexes] = scpl.labels[k]
            indexes.append(random_indexes)

        lp.fit(X,y)

        predict = lp.transduction_
        y = np.delete(predict, indexes).astype(int)
        y_true = np.delete(scpl.y_true, indexes)

        lp_accuracy = np.diag(confusion_matrix(y_true,y)*100/(counts-seeds[i]))

        y = -np.ones(shape=len(scpl.Y))

        acc[1,:,j] = lp_accuracy

        """ Label Spreading """
        ls = LabelSpreading()

        indexes = []
        for k in range(len(scpl.labels)):
            label_indexes = np.where(scpl.y_true == scpl.labels[k])[0]
            random_indexes = np.random.choice(label_indexes, size=seeds[i], replace=False)
            y[random_indexes] = scpl.labels[k]
            indexes.append(random_indexes)

        ls.fit(X, y)

        predict = ls.transduction_

        y = np.delete(predict,indexes).astype(int)
        y_true = np.delete(scpl.y_true, indexes)

        ls_accuracy = np.diag(confusion_matrix(y_true,y)*100/(counts - seeds[i]))
        y = -np.ones(shape=len(scpl.Y))

        acc[2,:,j] = lp_accuracy
       
    accuracies_mean[:,:,i] = np.mean(acc, axis=2)
    accuracies_std[:,:,i] = np.std(acc, axis=2)

    diameters_mean[i] = np.mean(diam)
    diameters_std[i] = np.std(diam)

ngplot.compare_algs(accuracies_mean)
ngplot.compare_algs(accuracies_std)
"Do Not Show Percentage, 3 STDS"
ngplot.diameters_per_seeds(seeds, diameters_mean, show_value=False, stds=3, colors='b', std_array=diameters_std)
"Do Not Show Percentage, 2 STDS"
ngplot.diameters_per_seeds(seeds, diameters_mean, show_value=False, stds=2, colors='b', std_array=diameters_std)
"Do Not Show Percentage, 1 STDS"
ngplot.diameters_per_seeds(seeds, diameters_mean, show_value=False, stds=1, colors='b', std_array=diameters_std)