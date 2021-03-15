import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mpl_toolkits.axes_grid1 import ImageGrid

import sys
import os
path = os.getcwd()
sys.path.insert(0,path)

moons = np.load('data/Moons.npy')
clusters_09 = np.load('data/Cluster_09.npy')
clusters_12 = np.load('data/Cluster_12.npy')
classification = np.load('data/Classification.npy')
circles = np.load('data/Circles.npy')

seeds = np.array([2,5,10,12,15,20,22,25,30,32,35,40,42,45,50])
expand = np.array([2,3,4,5,10,20,30,40,50,75,100,200,300,400,500])
neighbors = np.array([5,10,15,20,25,50])

exp_ticks = np.linspace(2,500,num=8,dtype=int)
seed_ticks = np.linspace(2,50,num=8,dtype=int)
total = 400

moons = np.sum(moons,axis=1)
clusters_09 = np.sum(clusters_09,axis=1)
clusters_12 = np.sum(clusters_12,axis=1)
classification = np.sum(classification,axis=1)
circles = np.sum(circles,axis=1)

data = circles

fig, axes = plt.subplots(nrows=3,ncols=2,figsize=(6,8))
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.2, right=0.99)
for i, ax in enumerate(axes.flat):
    print(i)
    plot = ax.imshow(data[i]/total, cmap='rainbow')
    
    ax.set_xticks(np.arange(exp_ticks.size)*2)
    ax.set_yticks(np.arange(seed_ticks.size))
    ax.set_xticklabels(exp_ticks, fontsize=15)
    ax.set_yticklabels(seed_ticks, fontsize=15)
    if i < 4:
        ax.axes.get_xaxis().set_visible(False)
    if i % 2 != 0:
        ax.axes.get_yaxis().set_visible(False)
    ax.set_ylabel('Sementes', fontsize=20)
    ax.set_xlabel('Expansão', fontsize=20)
    ax.set_title('Vizinhos: {}'.format(neighbors[i]))
    fig.colorbar(plot, ax=ax)
plt.tight_layout(pad=4, w_pad=0, h_pad=2.0)
plt.show()
