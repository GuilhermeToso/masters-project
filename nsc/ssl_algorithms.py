""" Dependências """
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from scipy import sparse
from model import Preprocessing
import sys
import os

__all__=['PCC']

class PCC():

    def __init__(self, *args, **kwargs):

        self.preprocessing = Preprocessing()

        self.adjacency = kwargs.get('adjacency')
        self.categorical = kwargs.get('categorical')
        self.control = kwargs.get('control')
        self.data = kwargs.get('data')
        self.degree = kwargs.get('degree')
        self.delta = kwargs.get('delta')
        self.distance = kwargs.get('distance')
        self.domination = kwargs.get('domination')
        self.indexes_of = {}
        self.labels_list = kwargs.get('labels')
        self.maximum = kwargs.get('max')
        self.numerical = kwargs.get('numerical')
        self.particles = kwargs.get('particles')
        self.pgrd = kwargs.get('pgrd')
        self.positions = kwargs.get('positions')
        self.pot_min = kwargs.get('w_min')
        self.pot_max = kwargs.get('w_max')
        self.target = kwargs.get('target')
        self.threshold = kwargs.get('threshold')
        self.X = kwargs.get('X')
        self.Y = kwargs.get('Y')
        self.y_predicted = kwargs.get('y_predict')

    def define_adjacency(self,data_dtype, connection='threshold'):

        if connection == 'threshold':
            self.distance = np.linalg.norm(data_dtype-data_dtype[:,None,:],axis=2)
            self.adjacency = np.where(self.distance<self.threshold,1,0)
            np.fill_diagonal(self.adjacency,0)
        elif connection == 'degree':
            self.distance = np.linalg.norm(data_dtype-data_dtype[:,None,:],axis=2)
            columns = self.distance.argsort(axis=1)[:,:self.degree].flatten()
            rows = np.repeat(np.arange(self.data.shape[0]),self.degree)
            self.adjacency = np.zeros(shape=(self.data.shape[0], self.data.shape[0]))
            self.adjacency[rows,columns] = 1
    def initial_configuration(self):

        particles = 0
        self.positions = np.array([])
        for label in self.labels_list:
            particles = particles + self.indexes_of['label'][label].size
            self.positions = np.append(self.positions,self.indexes_of['label'][label])
        self.domination = np.zeros(shape=(self.data.shape[0],len(self.labels_list))) + self.pot_min
        self.particles = np.zeros(shape=(particles,3))
        
        self.particles[:,2] = self.positions.astype(int)
        self.target = self.positions.astype(int)

        initial = 0
        for i in range(len(self.labels_list)):
            final = initial + self.indexes_of['label'][self.labels_list[i]].size
            self.domination[self.indexes_of['label'][self.labels_list[i]],i] = self.pot_max
            self.particles[initial:final,1] = self.labels_list[i]
            initial = final
        set_diff = np.setdiff1d(np.arange(self.data.shape[0]),self.positions)
        self.domination[set_diff,:] =self.pot_min + (self.pot_max-self.pot_min)/len(self.labels_list)
        self.particles[:,0] = 1

        self.min = np.zeros(shape=len(self.labels_list))+self.pot_min


        self.particles_distance = np.zeros(shape=(particles,self.data.shape[0])) + self.data.shape[0] - 1
        self.particles_distance[np.arange(self.particles.shape[0]),self.particles[:,2].astype(int)] = 0

    def move(self,i):

       
        self.random = np.random.uniform(0,1)

        if self.random < self.pgrd:
            
            adjacency = self.adjacency[self.target[i].astype(int),:]
            self.group_strengh = self.domination[:,self.particles[i,1].astype(int)]
            self.dist_inverse = 1/(1 + self.particles_distance[i,:])**2
            self.prob = adjacency*self.group_strengh*self.dist_inverse
            self.greedy_walk = self.prob/np.sum(self.prob)
            self.indexes = np.where(self.greedy_walk!=0)[0]
            self.accumulated = np.cumsum(self.greedy_walk[self.indexes])
            self.generated = np.random.uniform(0,1)
            self.found = np.where(self.accumulated>self.generated)[0][0]
            self.chosen = self.indexes[self.found]

        else:
            random_vector = self.adjacency[self.target[i].astype(int),:]/np.sum(self.adjacency[self.target[i].astype(int),:])
            self.indexes = np.where(random_vector!=0)[0]
            self.accumulated = np.cumsum(random_vector[self.indexes])
            self.generated = np.random.uniform(0,1)
            self.found = np.where(self.accumulated>self.generated)[0][0]
            self.chosen = self.indexes[self.found]

        return self.chosen


    def update(self,i,target):

        
        
        if target not in self.positions:
            self.label = self.particles[i][1].astype(int)
            domination_decrease = self.domination[int(target),:] - np.maximum(self.min,self.domination[int(target),:] - self.delta*self.particles[i,0]/(len(self.labels_list)-1))
            self.domination[int(target),:] = self.domination[int(target),:] - domination_decrease
            self.domination[int(target),self.label] = self.domination[int(target),self.label] + np.sum(domination_decrease) 
            
            self.particles[i,0] = self.particles[i,0] + (self.domination[int(target),self.label] - self.particles[i,0])*self.control

        distance = self.particles_distance[i,self.particles[i,2].astype(int)] + 1
        if distance < self.particles_distance[i,int(target)]:
            self.particles_distance[i,int(target)] = distance

        self.particles[i,2] = target

    def fit(self, epochs, data_dtype):

        self.y_predicted = -np.ones(shape=(self.data.shape[0]))

        self.define_adjacency(data_dtype)
        self.initial_configuration()
        self.target = -np.ones(shape=(self.particles.shape[0]))
        max_domination = 0
        i = 0
        while max_domination < self.maximum and i < epochs:
            print("Iteration: ", i)
            for j in range(self.particles.shape[0]):
                self.target[j] = self.move(j)
                self.update(j,self.target[j])
            max_domination = np.mean(self.domination.max(axis=1))
            i = i+1
            print("Domination mean: ", max_domination)
        self.y_predicted = np.argmax(self.domination,axis=1)
        print('Confusion Matrix: ', confusion_matrix(self.Y, self.y_predicted))


    def preprocess_data(self, shuffle = True, split = True, set_null = True, not_null = None, get_indexes = True):
        
        self.labels_list = self.preprocessing.get_labels(self.data, self.target)

        if shuffle == True:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        if split == True:
            self.X, self.categorical, self.numerical, self.Y = self.preprocessing.split_data(self.data, self.target)
        if set_null == True:

            if isinstance(not_null, int):

                self.preprocessing.set_null_values(self.data, self.target, self.labels_list,label_size=not_null)
            
            elif isinstance(not_null, dict):
                
                self.preprocessing.set_null_values(self.data, self.target, label_dict=not_null)
        
        if get_indexes == True: 

            self.indexes_of['unlabel'], self.indexes_of['label'] = self.preprocessing.get_label_unlabel_inds(self.data, self.target, self.labels_list)               
            self.indexes_of['unlabel'] = np.array(self.indexes_of['unlabel'])
        print("\n-------------------The data has been preprocessed --------------------\n")
    