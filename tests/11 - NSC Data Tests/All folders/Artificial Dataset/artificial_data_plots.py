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
from model import NeuronGraph
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from model import ngplot
from model import unwrap
from tqdm import tqdm
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from matplotlib.colors import ListedColormap
np.random.seed(0)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams['text.usetex'] = True
import multiprocessing

def concat_data(x,y):

    x_pd = pd.DataFrame(x,columns=['X','Y'])
    y_pd = pd.DataFrame(y,columns=['target'])

    df = pd.concat([x_pd,y_pd],axis=1)

    return df


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

data = [moons, clusters_09, clusters_12, classification, circles]
names = ['Luas', r'Clusters - $\sigma=0.9$', r'Clusters - $\sigma=1.2$', 'Classificação', 'Círculos']

def plot_data(data,names):

    cols = ['r','b','g']
    fig, axes = plt.subplots(nrows=3,ncols=2,figsize=(6,8))
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.2, right=0.99)
    for i, ax in enumerate(axes.flat):

        if i <= 4:
            num = data[i]['target'].unique().size
            for j in range(num):
                plot = ax.scatter(data[i]['X'].loc[data[i]['target']==j],\
                    data[i]['Y'].loc[data[i]['target']==j], s=50,c=cols[j])
            
            ax.set_ylabel('Y', fontsize=20)
            ax.set_xlabel('X', fontsize=20)
            ax.tick_params(labelsize=15)
            # if i < 4:
            #     ax.axes.get_xaxis().set_visible(False)
            # if i % 2 != 0:
            #     ax.axes.get_yaxis().set_visible(False)
            ax.set_title(names[i], fontsize=15)
        else:
            ax.set_visible(False)
    plt.tight_layout(pad=4, w_pad=0, h_pad=2.0)
    plt.show()

iris = datasets.load_iris(as_frame=True).frame
print(iris.columns)

data = iris
print(data['target'])

model = 'CNV'
model_1 = 'CNV'
thresh = 0.2

ng = NeuronGraph(data=data, target='target', similarity='Euclidean',
model=model, alpha = 0.1, w_step = 0.2, time_step=1, print_info=False,
print_steps=False, beta=2.0, gamma=1.5)
ng.neighbors = 15
ng.search_expand = 100
ng.preprocess_data(shuffle=False,not_null=10,standarlize=False)
print(ng.Y)
epochs = 1000
ng.fit(epochs,ng.numerical)
diag =  np.diagonal(ng.confusion_matrix)
if diag.size==len(ng.labels_list):
    score = diag.sum()/data.shape[0]
elif diag.size > len(ng.labels_list):
    score = diag[-len(ng.labels_list):].sum()/data.shape[0]
print("Score: {}", round(score,2))

x_min, x_max = data['sepal length (cm)'].min(), data['sepal length (cm)'].max()
y_min, y_max = data['petal length (cm)'].min(), data['petal length (cm)'].max()

delta_x = abs(x_max-x_min)
delta_y = abs(y_max-y_min)
if delta_y>delta_x:
    h = delta_x/10
else:
    h = delta_y/10

xx, yy = np.meshgrid(np.arange(x_min,x_max,h),\
    np.arange(y_min,y_max,h))

test_array = np.c_[xx.ravel(),yy.ravel()]

predicted = ng.predict(test_array,ng.numerical[:,[0,2]])
predicted = predicted.reshape(xx.shape)

cm = plt.cm.RdBu
if len(ng.labels_list) == 3:
    cm_bright = ListedColormap(['b','g','r'])
else:
    cm_bright = ListedColormap(['g','r'])
fig,ax = plt.subplots()
ax.contourf(xx, yy, predicted, cmap='rainbow', alpha=.8)
ax.scatter(data['sepal length (cm)'],data['petal length (cm)'], c=ng.y_predicted,cmap=cm_bright, edgecolor='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
#ax.set_xticks(())
#ax.set_yticks(())
ax.tick_params(labelsize=30)
ax.set_xlabel('Comprimento da Sépala (cm)', fontsize=30,labelpad=30)
ax.set_ylabel('Comprimento da Pétala (cm)', fontsize=30,labelpad=30)
plt.show()


ngplot.trajectories(ng.neuron.potential[:,0,:],np.arange(epochs)*ng.time_step)
# ngplot.trajectories(ng.neuron.potential[:,0,ng.indexes_of['label'][0]],np.arange(epochs)*ng.time_step)
# ngplot.trajectories(ng.neuron.potential[:,0,ng.indexes_of['label'][1]],np.arange(epochs)*ng.time_step)
# ngplot.trajectories(ng.neuron.potential[:,0,ng.indexes_of['label'][2]],np.arange(epochs)*ng.time_step)

""" Get the peak indexes, times and the periods between them """
inds, times, pers = unwrap.get_peaks_indexes(ng.neuron.potential[:,0,:], thresh, ng.time_step)
#print(inds)

""" Get the phases """
phases = unwrap.unwrap_static_2(ng.neuron.potential.shape[0], inds, ng.time_step, model=model_1)


neurons_array = []
for i in range(len(times)):
    neurons_array.append(np.zeros(times[i].size)+i)
ngplot.neural_activity(times, neurons_array,epochs*ng.time_step)
# print(ng.indexes_of['label'][0])
# print(ng.indexes_of['label'][0][0])

# print(ng.indexes_of['label'][1])
# print(ng.indexes_of['label'][1][0])
# print(ng.indexes_of['label'][2])
# print(ng.indexes_of['label'][2][0])

""" Get the Phases difference  with group1 as reference"""
ngplot.phases_diff_3D(int(ng.indexes_of['label'][0][0]), phases, ng.time_step)

""" Get the Phases difference  with group2 as reference"""
ngplot.phases_diff_3D(int(ng.indexes_of['label'][1][0]), phases, ng.time_step)

""" Get the Phases difference  with group2 as reference"""
ngplot.phases_diff_3D(int(ng.indexes_of['label'][2][0]), phases, ng.time_step)
