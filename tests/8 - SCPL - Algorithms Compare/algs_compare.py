import sys
import os
path = os.getcwd()
sys.path.insert(0,path)
print(path)
from nsc import unwrap
from nsc import ngplot
from nsc import SCPL
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
#from pandas.compat import StringIO
from sklearn.preprocessing import LabelEncoder
from nsc import TriTraining, StandardSelfTraining
import sklearn.datasets as ds


DATASETS = ['iris', 'bcw', 'wine', 'bank']

dataset = {'iris':ds.load_iris(as_frame=True).frame}
for i in range(len(DATASETS)-1):
    dataset[DATASETS[i+1]] = pd.read_csv(path+'/data/cleaned/'+DATASETS[i+1]+'.csv')

iris = dataset['iris']
bcw = dataset['bcw']
wine = dataset['wine']
bank = dataset['bank']


def split_data(data):
    y_train = data['target'].to_numpy()
    x_train = data.drop(['target'],axis=1).to_numpy()
    y_true = y_train.copy()
    return x_train,y_train,y_true

def choose_labeled(y_true, num):
    labels = np.unique(y_true)
    y_train = y_true.copy()
    for i in range(labels.size):
        where = np.where(y_true==labels[i])[0]
        np.random.shuffle(where)
        y_train[where[:where.size-num]]=-1
    return y_train


def get_scores(dataset, name):

    seeds = np.arange(5,20)
    if name != 'iris':
        seeds = np.arange(1,15)*15
    
    data = dataset[name]

    scpl = SCPL(data=data.copy(), target='target',similarity='Euclidean')
    lp = LabelPropagation()
    ls = LabelSpreading()
    KNN = KNeighborsClassifier(
        n_neighbors=3,
        metric='euclidean',
        #n_jobs=2  # Parallelize work on CPUs
    )
    sst_knn = StandardSelfTraining("Self-Training (KNN)",KNN)
    tt_knn = TriTraining("Tri-Training (KNN)",KNN)
    
    NB = GaussianNB(
    priors=None
    )
    sst_nb = StandardSelfTraining("Self-Training (NB)",NB)
    tt_nb = TriTraining("Tri-Training (NB)",NB)
    
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
    tt_dt = TriTraining("Tri-Training (DT)",DT)
    
    SVM = SVC(
    C=1.0,
    kernel='poly',
    degree=1,
    tol=0.001,
    probability=True
    ) 
    sst_svm = StandardSelfTraining("Self-Training (SVM)",SVM)
    tt_svm = TriTraining("Tri-Training (SVM)",SVM)
    

    scores = np.zeros(shape=(seeds.size, 50, 11))

    for i in tqdm(range(seeds.size)): 

        for j in tqdm(range(50)):
            
            """ SCPL ALGORITHM """
            scpl.data_process(set_null=True, label_size=int(seeds[i]))            
            y_train = -np.ones(shape=(data.shape[0])).astype(int)
            y_true = scpl.y_true

            for key, value in scpl.labels_indexes.items():
                y_train[value] = key
            scpl.fit(scpl.numerical)
            scpl.data = data.copy()
            scores[i,j,0] = scpl.score

            x_train = scpl.numerical


            """ LABEL PROPAGATION """
            lp.fit(x_train,y_train)
            scores[i,j,1] = lp.score(x_train,y_true)

            """ LABEL SPREADING """
            ls.fit(x_train,y_train)
            scores[i,j,2] = ls.score(x_train,y_true)

            y_train_str = y_train.astype(str)
            y_train_str[y_train_str=='-1']='unlabeled'
            y_true_str=y_true.astype(str)

            """ SST+KNN """
            sst_knn.fit(x_train,y_train_str)
            scores[i,j,3] = sst_knn.score(x_train,y_true_str)
        
            """ SST+NB """
            
            sst_nb.fit(x_train,y_train_str)
            scores[i,j,4] = sst_nb.score(x_train,y_true_str)
        
            """ SST+DT """
            sst_dt.fit(x_train,y_train_str)
            scores[i,j,5] = sst_dt.score(x_train,y_true_str)
        
            """ SST+SVM """
            sst_svm.fit(x_train,y_train_str)
            scores[i,j,6] = sst_svm.score(x_train,y_true_str)
        
            """ TT+KNN """
            tt_knn.fit(x_train,y_train_str)
            scores[i,j,7] = tt_knn.score(x_train,y_true_str)
        
            """ TT+NB """
            tt_nb.fit(x_train,y_train_str)
            scores[i,j,8] = tt_nb.score(x_train,y_true_str)
        
            """ TT+DT """
            tt_dt.fit(x_train,y_train_str)
            scores[i,j,9] = tt_dt.score(x_train,y_true_str)
        
            """ TT+SVM """
            tt_svm.fit(x_train,y_train_str)
            scores[i,j,10] = tt_svm.score(x_train,y_true_str)
        
    np.save('Scores_'+name,scores)

    return scores

gs = get_scores(dataset,'wine')
print(gs.mean(axis=1))
print(gs.std(axis=1))