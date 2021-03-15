# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 20:34:53 2020

@author: User
"""

import sys
import os
import platform

system = platform.system()

current_dir = os.getcwd()

if system == 'Windows':
    path_dir = current_dir.split("\\Neurons")[0] + "\\Neurons"
else:
    path_dir = current_dir.split("/Neurons")[0] + "/Neurons"

sys.path.append(path_dir)
#from model import TriTraining, StandardSelfTraining,SKTSVM
from model import Preprocessing
#from sklearn.datasets.mldata import fetch_mldata
from sklearn.linear_model import SGDClassifier
import sklearn.svm
import pandas as pd
import numpy as np
import sklearn
print(sklearn.__version__)
from scipy.spatial.distance import hamming


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

print(mpl.__version__)
sys.exit(0)
from model import ngplot
from model import unwrap
from tqdm import tqdm

import sklearn.datasets as ds
from sklearn.metrics import confusion_matrix
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from pandas.compat import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# Path
path = r"C:\\Users\\Avell\\Trabalho\\Mestrado\\Projeto\\Neurons\\Tests\\14 - NeuronGraph Data Tests\\Data-tests\\comparison\\datas\\"

def spam(path):

    spambase = pd.read_csv(path+'spambase.data')

    # Get header
    new_header = spambase.columns.values

    # Create the list of the new columns' names
    cols = ['word_freq_'+str(i) for i in range(48)]
    cols = cols + ['char_freq_'+str(i) for i in range(6)]
    cols = cols + ['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'target']

    # Rename the spambase columns
    spambase = spambase.rename(columns={new_header[i]:cols[i] for i in range(len(cols))})

    # Create a dataframe with one row, which is the same as the header
    sb = pd.DataFrame(new_header[None,:],columns=list(cols))

    # Fix some data
    new_header[2] = '0.64'
    new_header[11] = '0.64'
    new_header[15] = '0.32'
    new_header[40] = '0.32'
    # Transform the data to float
    sb.loc[0,cols] = new_header.astype(float)
    sb = sb.astype(float)

    # The last three transform to integer
    sb['capital_run_length_longest'] = sb['capital_run_length_longest'].astype('int64')
    sb['capital_run_length_total'] = sb['capital_run_length_total'].astype('int64')
    sb['target'] = sb['target'].astype('int64')

    # Append both data frames
    sb =sb.append(spambase)

    # Now let's analyse the information and description
    print(sb.info())
    # There are no missing data

    # See the description
    print(sb.describe())
    # The data variate to much, so we will standarlize it

    # Standard Data
    scaler = StandardScaler()
    sb.loc[:,cols[:-1]] = scaler.fit_transform(sb.loc[:,cols[:-1]])

    # Now let's see the amount of samples per classes
    count = sb['target'].value_counts().rename_axis('Unique').to_frame('Counts')
    plt.bar(np.arange(1),count['Counts'][0],color='b',alpha=0.5, label='Not Spam')
    plt.bar(np.arange(1)+1,count['Counts'][1],color='r',alpha=0.5, label='Spam')
    plt.xlabel('Classes',fontsize=20, labelpad=30)
    plt.ylabel('Frequency',fontsize=20, labelpad=30)
    plt.xticks(np.arange(2),['0','1'],fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend()
    plt.show()
    print("Not Spam: {}%".format(round(count['Counts'][0]*100/count['Counts'].sum(),2)))
    print("Spam: {}%".format(round(count['Counts'][1]*100/count['Counts'].sum(),2)))
    print("Ratio Not Spam/Spam: {}%".format(round((count['Counts'][0]/count['Counts'][1])*100,2)))
    sb.to_csv(path+'spam_base.data')

#def diabetes(path):

def diab(path):

    diabetes = pd.read_csv(path+'pima-indians-diabetes.csv')
    print(diabetes)
    sys.exit(0)
    new_header = diabetes.columns.values

    cols = ['n_times_pregnant','plasma_gluc_conc','diastolic_blood_conc', \
        'skin_thick','2-hour_serum_insulin','body_mass_ind','diabetes_pedigree_f','age','target']

    diabetes = diabetes.rename(columns={new_header[i]:cols[i] for i in range(len(cols))})
    diabetes = diabetes.astype('float64')
    diabetes['target'] = diabetes['target'].astype('int64')


    db = pd.DataFrame(new_header.astype('float64')[None,:], columns=cols)
    db['target'] = db['target'].astype('int64')

    db = db.append(diabetes)

    # Let's Standarlize
    scaler = StandardScaler()
    db.loc[:,cols[:-1]] = scaler.fit_transform(db.loc[:,cols[:-1]])

    db.to_csv(path+'diabetes.data')

def sonar_data(path):

    sonar = pd.read_csv(path+'sonar.all-data')
    print(sonar)
    sys.exit(0)
    new_header = sonar.columns.values
    cols = ['E_'+str(i+1) for i in range(60)]
    cols = cols + ['target']

    sonar = sonar.rename(columns={new_header[i]:cols[i] for i in range(len(cols))})
    le = LabelEncoder()
    sonar['target'] =le.fit_transform(sonar['target'])


    s = pd.DataFrame(new_header[None,:], columns=cols)
    s['target'] = '1'
    s = s.astype('float64')
    s['target'] = s['target'].astype('int32')

    s = s.append(sonar)

    s.to_csv(path+'sonar.data')

bcw_1 = sklearn.datasets.load_breast_cancer(as_frame=True).frame

features = bcw_1.drop(['target'],axis=1)
print(features)

scaler = StandardScaler()
X = scaler.fit_transform(features)
pca = PCA()
pca_data = pca.fit_transform(X)
pca_var = pca.explained_variance_ratio_
var_cum = np.cumsum(pca_var)
var_limit = 0.95
index = np.where((var_cum >= var_limit))[0][0]
data = pca_data[:,:index]

df1 = pd.DataFrame(data, columns=['PC_'+str(i+1) for i in range(index)])
final_bcw = pd.concat([df1, bcw_1['target']], axis=1)
final_bcw.to_csv(path+'bcw.data')


# pca = PCA()
# result = pca.fit_transform(bcw.loc[:,bcw.columns.values[:-1]])
# bcw_pca = pd.DataFrame(result,columns=['pc_'+str(i+1) for i in range(16)])
# #print(bcw_pca)
# final_bcw = pd.concat([bcw_pca, bcw['target']], axis=1)
# final_bcw.to_csv(path+'bcw.data')

#n = sonar_data(path)
#print(n)
#print(bcw_1)
# print(arry.loc[:,arry_header[11:14]])
# print(arry_header)

wine = pd.read_csv('wine.data')
scaler = StandardScaler()
X = scaler.fit_transform(wine.loc[:,wine.columns.values[:-1]])
pca = PCA()
pca_data = pca.fit_transform(X)
pca_var = pca.explained_variance_ratio_
var_cum = np.cumsum(pca_var)
var_limit = 0.95
print(var_cum)
index = np.where((var_cum >= var_limit))[0][0]
data = pca_data[:,:index]

data = pd.DataFrame(data, columns=['PC_'+str(i+1) for i in range(index)])
print(data)
wine = pd.concat([data,wine['quality']], axis=1)
wine = wine.rename(columns={'quality':'target'})
print(wine)
wine.to_csv(path+'wine_quality_1.data')

#sonar = pd.read_csv('sonar.data')

a = np.array([1,0,0,1,2,3,5])
b = np.array([
    [1,0,3,4,3,2,5],
    [2,0,0,1,2,5,6],
    [3,3,0,1,2,4,4],
    [0,7,0,1,4,4,5] 
])

d = (a.size - np.sum(a==b, axis=1))/a.size
print(d)