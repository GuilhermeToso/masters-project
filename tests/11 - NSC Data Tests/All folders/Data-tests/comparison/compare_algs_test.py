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
from model import TriTraining, StandardSelfTraining, NeuronGraph
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

path = r"C:\\Users\Avell\\Trabalho\\Mestrado\\Projeto\\Neurons\\Tests\\10 - NeuronGraph Data Tests\\Data-tests\\comparison\\"

DATASETS = ['banana','banknote','bcw','diabetes','nursery_enc',\
    'nursery_ohe','phoneme', 'sonar','spam_base','titanic','wine_quality']

TARGETS = {
    'banana': {1:'Forma_1',-1:'Forma_2'}, 'banknote':{0:'Authentic',1:'Inauthentic'},\
        'bcw':{0:'Malignant',1:'Benign'}, 'diabetes':{0:'Negative',1:'Positive'},\
        'nursery_enc':{0:'not_recom',1:'priority',2:'spec_prior',3:'very_recom'},\
        'nursery_ohe':{0:'not_recom',1:'priority',2:'spec_prior',3:'very_recom'},\
        'phoneme':{0:'nasal',1:'oral'}, 'sonar':{0:'M',1:'R'}, 'spam_base':{0:'not_spam',1:'spam'},\
            'titanic':{0:'No',1:'Yes'},'wine_quality':{i+3:str(i+3) for i in range(7)}
}

dataset = {}
for i in range(len(DATASETS)):
    dataset[DATASETS[i]] = pd.read_csv(path+DATASETS[i]+'.data')

dataset['banknote'] = dataset['banknote'].drop(['Unnamed: 0'],axis=1)
dataset['bcw'] = dataset['bcw'].drop(['Unnamed: 0'],axis=1)
dataset['diabetes'] = dataset['diabetes'].drop(['Unnamed: 0'],axis=1)
dataset['nursery_enc'] = dataset['nursery_enc'].drop(['Unnamed: 0'],axis=1)
dataset['nursery_ohe'] = dataset['nursery_ohe'].drop(['Unnamed: 0'],axis=1)
dataset['phoneme'] = dataset['phoneme'].drop(['Unnamed: 0'],axis=1)
dataset['sonar'] = dataset['sonar'].drop(['Unnamed: 0'],axis=1)
dataset['spam_base'] = dataset['spam_base'].drop(['Unnamed: 0'],axis=1)
dataset['titanic'] = dataset['titanic'].drop(['Unnamed: 0'],axis=1)
dataset['wine_quality'] = dataset['wine_quality'].drop(['Unnamed: 0'],axis=1)
dataset['wine_quality'] = dataset['wine_quality'].drop(['Unnamed: 0.1'],axis=1)

""" 
MODELS:
    lp - Label Propagation
    ls - Label Spreading
    sst + c - Self Standard Training + Classifier
    tt + c - Tri-Training + Classifier
    ng - NeuronGraph
 """


bank = dataset['banknote']       # Continuous - lp('rbf'), ls('rbf')
banana = dataset['banana']       # Continuous
bcw = dataset['bcw']             # Continuous
diabetes = dataset['diabetes']   # Continuous
nur_enc = dataset['nursery_enc'] # Categorial
nur_ohe = dataset['nursery_ohe'] # Continuous One Hot Enc
phoneme = dataset['phoneme']     # Continuous
sonar = dataset['sonar']         # Continuous
spam = dataset['spam_base']      # Continuous
titanic = dataset['titanic']     # Categorical
wine = dataset['wine_quality']   # Continuous]
wine.loc[:,'target'] = LabelEncoder().fit_transform(wine['target'])

print(phoneme)
# wine_1 = pd.read_csv(path+'wine_quality_1.data')
# print(wine_1)
# wine_1.loc[:,'target'] = LabelEncoder().fit_transform(wine_1['target'])
# print("bcw shape : ", bcw.shape)

# print("bcw: ", bcw)
# print("banana : ", banana.shape)
#print("diabetes : ", diabetes.shape)
#print(diabetes)
# print("spam : ", spam.shape)
# print("titanic : ", titanic.shape)
#sys.exit(0)

print(wine['target'].value_counts())
sys.exit(0)
def split_data(data):
    y_train = data['target'].to_numpy()
    x_train = data.drop(['target'],axis=1).to_numpy()
    y_true = y_train.copy()
    return x_train,y_train,y_true


def choose_labeled(y_true, num):
    labels = np.unique(y_true)
    y_train = y_true.copy()
    for i in range(labels.size):
        where = np.where(y_true==i)[0]
        np.random.shuffle(where)
        y_train[where[:where.size-int(num/labels.size)+1]]=-1
    return y_train


def data_lp(data):

    num = 10
    lp = LabelPropagation()
    scores = np.zeros(shape=(num,))
    cohen_kappa = np.zeros(shape=(num,))
    for i in range(num):
        x_train, y_train, y_true = split_data(data)
        y_train = choose_labeled(y_true,int(y_true.size*0.1))
        lp.fit(x_train,y_train)
        y_pred = lp.predict(x_train)
        scores[i] = lp.score(x_train,y_true)
        cohen_kappa[i] = cohen_kappa_score(y_pred,y_true)
    return scores, cohen_kappa

def data_ls(data):

    num = 10
    ls = LabelSpreading()
    scores = np.zeros(shape=(num,))
    cohen_kappa = np.zeros(shape=(num,))
    for i in range(num):
        x_train, y_train, y_true = split_data(data)
        y_train = choose_labeled(y_true,int(y_true.size*0.1))
        ls.fit(x_train,y_train)
        y_pred = ls.predict(x_train)
        scores[i] = ls.score(x_train,y_true)
        cohen_kappa[i] = cohen_kappa_score(y_pred,y_true)
    return scores, cohen_kappa

def data_sst_knn(data, metric='euclidean'):

    num = 10
    
    KNN = KNeighborsClassifier(
        n_neighbors=3,
        metric=metric,
        #n_jobs=2  # Parallelize work on CPUs
    )
    sst_knn = StandardSelfTraining("Self-Training (KNN)",KNN)
    
    scores = np.zeros(shape=(num,))
    cohen_kappa = np.zeros(shape=(num,))
    for i in range(num):
        x_train, y_train, y_true = split_data(data)
        y_train = choose_labeled(y_true,int(y_true.size*0.1))
        y_train = y_train.astype(str)
        y_train[y_train=='-1']='unlabeled'
        sst_knn.fit(x_train,y_train)
        y_pred = sst_knn.predict(x_train)
        scores[i] = sst_knn.score(x_train,y_true.astype(str))
        cohen_kappa[i] = cohen_kappa_score(y_pred.astype(float),y_true.astype(float))
    return scores, cohen_kappa

def data_sst_nb(data):

    num = 10
        
    NB = GaussianNB(
    priors=None
    )
    sst_nb = StandardSelfTraining("Self-Training (NB)",NB)
    
    scores = np.zeros(shape=(num,))
    cohen_kappa = np.zeros(shape=(num,))
    for i in range(num):
        x_train, y_train, y_true = split_data(data)
        y_train = choose_labeled(y_true,int(y_true.size*0.1))
        y_train = y_train.astype(str)
        y_train[y_train=='-1']='unlabeled'
        sst_nb.fit(x_train,y_train)
        y_pred = sst_nb.predict(x_train)
        scores[i] = sst_nb.score(x_train,y_true.astype(str))
        cohen_kappa[i] = cohen_kappa_score(y_pred.astype(float),y_true.astype(float))
    return scores, cohen_kappa

def data_sst_dt(data):

    num = 10
            
    DT = DecisionTreeClassifier(
            criterion='entropy',
            # splitter='best',
            # max_depth=None,
            # min_samples_split=2,
            min_samples_leaf=2,
            # min_weight_fraction_leaf=0.0,
            # max_features=None,
            # random_state=None,
            # max_leaf_nodes=None,
            # min_impurity_split=1e-07,
            # class_weight=None,
            # presort=False,
        )
    sst_dt = StandardSelfTraining("Self-Training (DT)",DT)
    
    scores = np.zeros(shape=(num,))
    cohen_kappa = np.zeros(shape=(num,))
    for i in range(num):
        x_train, y_train, y_true = split_data(data)
        y_train = choose_labeled(y_true,int(y_true.size*0.1))
        y_train = y_train.astype(str)
        y_train[y_train=='-1']='unlabeled'
        sst_dt.fit(x_train,y_train)
        y_pred = sst_dt.predict(x_train)
        scores[i] = sst_dt.score(x_train,y_true.astype(str))
        cohen_kappa[i] = cohen_kappa_score(y_pred.astype(float),y_true.astype(float))
    return scores, cohen_kappa

def data_sst_svm(data):

    num = 10
            
    SVM = SVC(
    C=1.0,
    kernel='poly',
    degree=1,
    tol=0.001,
    probability=True
    )
    
    sst_svm = StandardSelfTraining("Self-Training (SVM)",SVM)
    
    scores = np.zeros(shape=(num,))
    cohen_kappa = np.zeros(shape=(num,))
    for i in range(num):
        x_train, y_train, y_true = split_data(data)
        y_train = choose_labeled(y_true,int(y_true.size*0.1))
        y_train = y_train.astype(str)
        y_train[y_train=='-1']='unlabeled'
        sst_svm.fit(x_train,y_train)
        y_pred = sst_svm.predict(x_train)
        scores[i] = sst_svm.score(x_train,y_true.astype(str))
        cohen_kappa[i] = cohen_kappa_score(y_pred.astype(float),y_true.astype(float))
    return scores, cohen_kappa

def data_sst_cat(data):

    num = 10
            
    CAT = CategoricalNB()
    
    sst_cat = StandardSelfTraining("Self-Training (CAT)",CAT)
    
    scores = np.zeros(shape=(num,))
    cohen_kappa = np.zeros(shape=(num,))
    for i in range(num):
        x_train, y_train, y_true = split_data(data)
        y_train = choose_labeled(y_true,int(y_true.size*0.1))
        y_train = y_train.astype(str)
        y_train[y_train=='-1']='unlabeled'
        sst_cat.fit(x_train,y_train)
        y_pred = sst_cat.predict(x_train)
        scores[i] = sst_cat.score(x_train,y_true.astype(str))
        cohen_kappa[i] = cohen_kappa_score(y_pred.astype(float),y_true.astype(float))
    return scores, cohen_kappa

def data_tt_knn(data, metric='euclidean'):

    num = 10
    
    KNN = KNeighborsClassifier(
        n_neighbors=3,
        metric=metric,
        #n_jobs=2  # Parallelize work on CPUs
    )
    tt_knn = TriTraining("Tri-Training (KNN)",KNN)
    
    scores = np.zeros(shape=(num,))
    cohen_kappa = np.zeros(shape=(num,))
    for i in range(num):
        print("Iter: {}".format(i))
        x_train, y_train, y_true = split_data(data)
        y_train = choose_labeled(y_true,int(y_true.size*0.1))
        y_train = y_train.astype(str)
        y_train[y_train=='-1']='unlabeled'
        tt_knn.fit(x_train,y_train)
        y_pred = tt_knn.predict(x_train)
        scores[i] = tt_knn.score(x_train,y_true.astype(str))
        cohen_kappa[i] = cohen_kappa_score(y_pred.astype(float),y_true.astype(float))
    return scores, cohen_kappa

def data_tt_nb(data):

    num = 10
        
    NB = GaussianNB(
    priors=None
    )
    tt_nb = TriTraining("Tri-Training (NB)",NB)
    
    scores = np.zeros(shape=(num,))
    cohen_kappa = np.zeros(shape=(num,))
    for i in range(num):
        print("Iter: {}".format(i))
        x_train, y_train, y_true = split_data(data)
        y_train = choose_labeled(y_true,int(y_true.size*0.1))
        y_train = y_train.astype(str)
        y_train[y_train=='-1']='unlabeled'
        tt_nb.fit(x_train,y_train)
        y_pred = tt_nb.predict(x_train)
        scores[i] = tt_nb.score(x_train,y_true.astype(str))
        cohen_kappa[i] = cohen_kappa_score(y_pred.astype(float),y_true.astype(float))
    return scores, cohen_kappa

def data_tt_dt(data):

    num = 10
            
    DT = DecisionTreeClassifier(
            criterion='entropy',
            # splitter='best',
            # max_depth=None,
            # min_samples_split=2,
            min_samples_leaf=2,
            # min_weight_fraction_leaf=0.0,
            # max_features=None,
            # random_state=None,
            # max_leaf_nodes=None,
            # min_impurity_split=1e-07,
            # class_weight=None,
            # presort=False,
        )
    tt_dt = TriTraining("Tri-Training (DT)",DT)
    
    scores = np.zeros(shape=(num,))
    cohen_kappa = np.zeros(shape=(num,))
    for i in range(num):
        print("Iter: {}".format(i))
        x_train, y_train, y_true = split_data(data)
        y_train = choose_labeled(y_true,int(y_true.size*0.1))
        y_train = y_train.astype(str)
        y_train[y_train=='-1']='unlabeled'
        tt_dt.fit(x_train,y_train)
        y_pred = tt_dt.predict(x_train)
        scores[i] = tt_dt.score(x_train,y_true.astype(str))
        cohen_kappa[i] = cohen_kappa_score(y_pred.astype(float),y_true.astype(float))
    return scores, cohen_kappa

def data_tt_svm(data):

    num = 10
            
    SVM = SVC(
    C=1.0,
    kernel='poly',
    degree=1,
    tol=0.001,
    probability=True
    )
    
    tt_svm = TriTraining("Tri-Training (SVM)",SVM)
    
    scores = np.zeros(shape=(num,))
    cohen_kappa = np.zeros(shape=(num,))
    for i in range(num):
        print("Iter: {}".format(i))
        x_train, y_train, y_true = split_data(data)
        y_train = choose_labeled(y_true,int(y_true.size*0.1))
        y_train = y_train.astype(str)
        y_train[y_train=='-1']='unlabeled'
        tt_svm.fit(x_train,y_train)
        y_pred = tt_svm.predict(x_train)
        scores[i] = tt_svm.score(x_train,y_true.astype(str))
        cohen_kappa[i] = cohen_kappa_score(y_pred.astype(float),y_true.astype(float))
    return scores, cohen_kappa

def data_tt_cat(data):
    
    num = 10
            
    CAT = CategoricalNB()
    
    tt_cat = TriTraining("Tri-Training (CAT)",CAT)
    
    scores = np.zeros(shape=(num,))
    cohen_kappa = np.zeros(shape=(num,))
    for i in range(num):
        x_train, y_train, y_true = split_data(data)
        y_train = choose_labeled(y_true,int(y_true.size*0.1))
        y_train = y_train.astype(str)
        y_train[y_train=='-1']='unlabeled'
        tt_cat.fit(x_train,y_train)
        y_pred = tt_cat.predict(x_train)
        scores[i] = tt_cat.score(x_train,y_true.astype(str))
        cohen_kappa[i] = cohen_kappa_score(y_pred.astype(float),y_true.astype(float))
    return scores, cohen_kappa

def data_nsc(data, metric='Euclidean'):

    num = 10
    nsc = NeuronGraph(
        data=data, target='target', similarity=metric,model='Izhikevic',
        alpha=0.1, w_step=0.3, time_step=0.5, print_info=False, print_steps=False,
        beta = 2.0, gamma=1.5
    )
    nsc.neighbors = 15

    labels = np.unique(data['target']).size
    seeds = int(data.shape[0]*0.1)
    epochs = data.shape[0] + 200

    nsc.search_expand = 70


    scores = np.zeros(shape=(num,))
    cohen_kappa = np.zeros(shape=(num,))

    for i in range(num):
        print("Iter: {}".format(i))

        nsc.preprocess_data(not_null=int(seeds/labels), standarlize=False)

        nsc.fit(epochs, nsc.numerical)

        diag =  np.diagonal(nsc.confusion_matrix)
        if diag.size==len(nsc.labels_list):
            scores[i] = diag.sum()/data.shape[0]
        elif diag.size > len(nsc.labels_list):
            scores[i] = diag[-len(nsc.labels_list):].sum()/data.shape[0]
        print("Score: {}", round(scores[i],2))
        cohen_kappa[i] = cohen_kappa_score(nsc.y_predicted,nsc.Y)

        nsc.data = data
                    
        nsc.y_predicted = -np.ones(shape=nsc.neurons)

        nsc.degree_out = np.zeros(shape=(nsc.neurons,len(nsc.labels_list)+1))
        nsc.labels_array = -np.ones(shape=(nsc.neurons,nsc.neurons))
        nsc.incident_degree = np.zeros(shape=(nsc.neurons,len(nsc.labels_list)))
        nsc.inner_degree = {}
        nsc.graph.vertexes = np.array([])
        nsc.disputed = np.array([])
        nsc.capacity = np.zeros(shape=(nsc.neurons))

    return scores, cohen_kappa

# score, kappa = data_sst_knn(wine_1)

# print("\nScore wine: ", round(score.mean(),3))
# print("Score Std wine: ", round(score.std(),4))
# print("Kappa wine: ", round(kappa.mean(),3))
# print("Kappa Std wine: ", round(kappa.std(),4))


#sys.exit(0)
df1 = spam.loc[spam['target']==0]
df1 = df1[0:int(.5*df1.shape[0])]
df2 = spam.loc[spam['target']==1]
df2 = df2[0:int(.5*df2.shape[0])]

print(df1.shape)
print(df2.shape)
df3 = pd.concat([df1,df2])
print(df3.shape)

score_spam, kappa_spam = data_tt_svm(df3)
score_bcw, kappa_bcw = data_tt_svm(bcw)
score, kappa = data_tt_svm(diabetes)


print("\nScore spam: ", round(score_spam.mean(),3))
print("Score Std spam: ", round(score_spam.std(),4))
print("Kappa spam: ", round(kappa_spam.mean(),3))
print("Kappa Std spam: ", round(kappa_spam.std(),4))



print("\nScore bcw: ", round(score_bcw.mean(),3))
print("Score Std bcw: ", round(score_bcw.std(),4))
print("Kappa bcw: ", round(kappa_bcw.mean(),3))
print("Kappa Std bcw: ", round(kappa_bcw.std(),4))


print("\nScore diabetes: ", round(score.mean(),3))
print("Score Std diabetes: ", round(score.std(),4))
print("Kappa diabetes: ", round(kappa.mean(),3))
print("Kappa Std diabetes: ", round(kappa.std(),4))




# KNN = KNeighborsClassifier(
#         n_neighbors=3,
#         metric="euclidean",
#         #n_jobs=2  # Parallelize work on CPUs
#     )
# NB = GaussianNB(
# priors=None
# )
# CART = DecisionTreeClassifier(
#         criterion='entropy',
#         # splitter='best',
#         # max_depth=None,
#         # min_samples_split=2,
#         min_samples_leaf=2,
#         # min_weight_fraction_leaf=0.0,
#         # max_features=None,
#         # random_state=None,
#         # max_leaf_nodes=None,
#         # min_impurity_split=1e-07,
#         # class_weight=None,
#         # presort=False,
#     )
# SVM = SVC(
#    C=1.0,
#    kernel='poly',
#    degree=1,
#    tol=0.001,
# )

# y_train = y_train.astype(str)
# y_train[y_train=='-1'] = 'unlabeled'
# print(y_train)
# sst_knn = StandardSelfTraining("Self-Training (KNN)",KNN)

# sst_knn.fit(x_train,y_train)
# y_pred = sst_knn.predict(x_train)
# print(confusion_matrix(y_true.astype(str),y_pred))

