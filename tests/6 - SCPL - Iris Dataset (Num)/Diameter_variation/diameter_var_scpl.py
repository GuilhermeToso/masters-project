""" 
Simplicial Complex Propagating Labels - Iris DataSet
===================================================

Analysis of the Threshold Value.
----------------------------------------------------

**Author**: Guilherme M. Toso
**Tittle**: diameter_var_scpl.py
**Project**: Semi-Supervised Learning Using Competition for Neurons' Synchronization


**Description**:

    This script uses the Simplicial Complex Propagating Labels Algorithm  to classify the unlabeled data
    and varies the threshold value between (6*std.max()/epochs) and 6*std.max().

"""

""" Dependencies """
import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
from nsc import HodgkinHuxley, SDE, Chemical
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
from scipy import sparse
import sklearn.datasets as datasets



""" Import the Data Set """

iris = datasets.load_iris(as_frame=True).frame

scpl = SCPL(data=iris, target='target', similarity='Euclidean')
scpl.data_process(set_null=True, label_size=5)
scpl.fit(scpl.numerical,epochs=100)

ngplot.diameters(scpl.range, scpl.threshold_list)