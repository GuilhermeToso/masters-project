import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn.datasets as datasets
import sys

h = .02
a = np.array([1.,4.,5.])
b = np.array([6,7,8])

x_min, x_max = a.min(), a.max()
y_min, y_max = b.min(), b.max()

xx, yy = np.meshgrid(np.arange(x_min,x_max,h),\
    np.arange(y_min,y_max,h))

print(xx)
print(yy)
b = np.c_[xx.ravel(), yy.ravel()]
print(b)
sys.exit(0)
b = np.where(a>5)
print(b)

np.random.seed(0)

def concat_data(x,y):

    x_pd = pd.DataFrame(x,columns=['X','Y'])
    y_pd = pd.DataFrame(y,columns=['target'])

    df = pd.concat([x_pd,y_pd],axis=1)

    return df


x_clusters_09,y_clusters_09 = datasets.make_blobs(n_samples=200,centers=3,n_features=2, cluster_std=0.9)
clusters_09 = concat_data(x_clusters_09,y_clusters_09)


x_min, x_max = clusters_09['X'].min(), clusters_09['X'].max()
y_min, y_max = clusters_09['Y'].min(), clusters_09['Y'].max()

xx, yy = np.meshgrid(np.arange(x_min,x_max,h),\
    np.arange(y_min,y_max,h))

z = clusters_09['target'].values
print(z)