
""" 
Simplicial Complex Propagating Labels - Iris DataSet
===================================================

Analysis of the Threshold Value and Accuracy per Labels Size.
------------------------------------------------------------

**Author**: Guilherme M. Toso
**Tittle**: scpl_acc_diam_per_labels.py
**Project**: Classificação Semi-Supervisionada utilizando Sincronização de Modelos de
         Osciladores por uma Força de Acoplamento

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
from nsc import ngplot
from nsc import SCPL
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn.datasets as datasets


""" Import the Data Set """
iris = datasets.load_iris(as_frame=True).frame


scpl = SCPL(data=iris, target='target',similarity='Euclidean')

seeds = np.arange(20).astype(int) + 1


accuracies_mean = np.zeros(shape=(seeds.size,3))
accuracies_std = np.zeros(shape=(seeds.size,3))
diameters_mean = np.zeros(shape=(seeds.size))
diameters_std = np.zeros(shape=(seeds.size))

y = iris['target']
for i in tqdm(range(seeds.size)):

    acc = np.zeros(shape=(100,3))
    diam = np.zeros(shape=(100))

    for j in range(100):
        print(j)
        scpl.data_process(set_null=True, label_size=seeds[i])
        scpl.fit(scpl.numerical)

        acc[j,:] = scpl.accuracy
        diam[j] = scpl.threshold_list[-1]

        scpl.data['target'] = y

    accuracies_mean[i,:] = np.mean(acc, axis=0)
    accuracies_std[i,:] = np.std(acc, axis=0)

    diameters_mean[i] = np.mean(diam)
    diameters_std[i] = np.std(diam)


accuracies_mean = accuracies_mean.T
accuracies_std = accuracies_std.T

""" No Y Limit, Show Percentage, 3 STDs """

ngplot.accuracy_per_seeds(seeds, accuracies_mean, scpl.labels, y_limit=False, stds=3, colors=['r','g','b'],std_array = accuracies_std)

""" No Y Limit, Show Percentage, 2 STDs """
ngplot.accuracy_per_seeds(seeds, accuracies_mean, scpl.labels, y_limit=False, stds=2, colors=['r','g','b'],std_array = accuracies_std)

""" No Y Limit, Show Percentage, 1 STDs """
ngplot.accuracy_per_seeds(seeds, accuracies_mean, scpl.labels, y_limit=False, stds=1, colors=['r','g','b'],std_array = accuracies_std)

""" No Y Limit, Don't Show Percentage, 3 STDs """

ngplot.accuracy_per_seeds(seeds, accuracies_mean, scpl.labels, y_limit=False, show_value=False, colors=['r','g','b'],std_array = accuracies_std)

""" No Y Limit, Don't Show Percentage, 2 STDs """
ngplot.accuracy_per_seeds(seeds, accuracies_mean, scpl.labels, y_limit=False, show_value=False, colors=['r','g','b'],std_array = accuracies_std)

""" No Y Limit, Don't Show Percentage, 1 STDs """
ngplot.accuracy_per_seeds(seeds, accuracies_mean, scpl.labels, y_limit=False, show_value=False, colors=['r','g','b'],std_array = accuracies_std)

""" Y Limit, Show Percentage, 3 STDs """

ngplot.accuracy_per_seeds(seeds, accuracies_mean, scpl.labels, stds=3, colors=['r','g','b'],std_array = accuracies_std)

""" Y Limit, Show Percentage, 2 STDs """
ngplot.accuracy_per_seeds(seeds, accuracies_mean, scpl.labels, stds=2, colors=['r','g','b'],std_array = accuracies_std)

""" Y Limit, Show Percentage, 1 STDs """
ngplot.accuracy_per_seeds(seeds, accuracies_mean, scpl.labels, stds=1, colors=['r','g','b'],std_array = accuracies_std)

""" Y Limit, Don't Show Percentage, 3 STDs """

ngplot.accuracy_per_seeds(seeds, accuracies_mean, scpl.labels, show_value=False, colors=['r','g','b'],std_array = accuracies_std)

""" Y Limit, Don't Show Percentage, 2 STDs """
ngplot.accuracy_per_seeds(seeds, accuracies_mean, scpl.labels, show_value=False, colors=['r','g','b'],std_array = accuracies_std)

""" Y Limit, Don't Show Percentage, 1 STDs """
ngplot.accuracy_per_seeds(seeds, accuracies_mean, scpl.labels, show_value=False, colors=['r','g','b'],std_array = accuracies_std)


"Do Not Show Percentage, 3 STDS"
ngplot.diameters_per_seeds(seeds, diameters_mean, show_value=False, stds=3, colors='b', std_array=diameters_std)
"Do Not Show Percentage, 2 STDS"
ngplot.diameters_per_seeds(seeds, diameters_mean, show_value=False, stds=2, colors='b', std_array=diameters_std)
"Do Not Show Percentage, 1 STDS"
ngplot.diameters_per_seeds(seeds, diameters_mean, show_value=False, stds=1, colors='b', std_array=diameters_std)