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

print(current_dir)
from model import TriTraining, StandardSelfTraining
from model import Preprocessing
#from sklearn.datasets.mldata import fetch_mldata
from sklearn.linear_model import SGDClassifier
import sklearn.svm
import pandas as pd
import numpy as np
import sklearn
print(sklearn.__version__)


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from model import ngplot
from model import unwrap
from tqdm import tqdm

import sklearn.datasets as ds
from sklearn.metrics import confusion_matrix
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from pandas.compat import StringIO
from sklearn.preprocessing import LabelEncoder

path = r"C:\\Users\Avell\\Trabalho\\Mestrado\\Projeto\\Neurons\\Tests\\14 - NeuronGraph Data Tests\\Data-tests\\comparison\\"



# banknote = pd.read_csv(path+'data_banknote_authentication.txt')
# banknote['0'].loc[banknote['0']==1] = 2
# banknote['0'].loc[banknote['0']==0] = 1
# banknote.to_csv(path+'banknote.csv', index=False)


# wine = pd.read_csv(path+'wine_quality.data')
# wine = wine.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)
# wine['target'] = LabelEncoder().fit_transform(wine.loc[:,['target']]) + 1
# print(wine)
# wine.to_csv(path+'wine.csv', index=False, header= False)

# sonar = pd.read_csv(path+'sonar.data')
# sonar = sonar.drop(['Unnamed: 0'],axis=1)
# sonar['target'] = sonar['target'] + 1
# print(sonar)
# sonar.to_csv(path+'sonar.csv', index=False, header= False)

# phoneme = pd.read_csv(path + 'phoneme.data')
# phoneme = phoneme.drop(['Unnamed: 0'],axis=1)
# phoneme['target'] = phoneme['target'] + 1
# print(phoneme)
# phoneme.to_csv(path+'phoneme.csv', index=False, header= False)

diabetes = pd.read_csv(path + 'diabetes.data')
diabetes = diabetes.drop(['Unnamed: 0'],axis=1)
diabetes['target'] = diabetes['target'] + 1

diabetes.to_csv(path+'diabetes.csv', index=False, header= False)

bcw = pd.read_csv(path + 'bcw.data')

bcw = bcw.drop(['Unnamed: 0'],axis=1)
bcw['target'] = bcw['target'] + 1

bcw.to_csv(path+'bcw.csv', index=False, header= False)

spam = pd.read_csv(path + 'spam_base.data')

spam = spam.drop(['Unnamed: 0'],axis=1)
spam['target'] = spam['target'] + 1
print(spam)
spam.to_csv(path+'spam.csv', index=False, header= False)
