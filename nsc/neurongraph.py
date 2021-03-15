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
import operator
from .similarities import euclidean, cosine, jaccard, manhattan, imed, hamming
from .preprocess import Preprocessing
from .neurons import *
import random
import sys
import os

""" Classe Couple """
class Couple():

    """ 
    .. py:class::
    
    This class allows the coupling phenomena between two or more oscillators

    .. py:function:: __init__(self,*args,**kwargs)

    :param data: The membrane potential values of the N neurons at time t
    :type data: numpy.ndarray
    
    """

    def __init__(self,*args,**kwargs):

        super().__init__()

        self.data = kwargs.get('data')
        
    def synapse(self, connections):

        """ 
        .. py::function:

        This function creates the synapse (coupling) between the neurons

        :param connections: A matrix of the coupling force between the neurons
        :type connections: numpy.ndarray
        
        """

        self.data_matrix = np.zeros(self.data.size) + self.data[:,np.newaxis]

        self.difference_data = self.data_matrix - self.data_matrix.T

        self.making_connections = self.difference_data*connections
    
        return np.mean(self.making_connections,axis=1)

""" Classe Modelo """
class Neuron(Couple):

    """ 
    .. py:class::

    This class represents the biological neuron model chosen, the amounts in the system and it's dynamics.

    .. py::function: __init__(self,*args,**kwargs)

    :param name: the neuron model name to be used.
    :type name: str
    :param model: the neuron model chosen
    :type model: NeuronType
    :param variables: the neuron model variables
    :type variables: list
    :param dynamic: the dynamic function of the neuron model.
    :type dynamic: function
    :param max_force: the maximum coupling force value. Default is None, depends on the model.
    :type max_force: int or float
    :param min_force: the minimum coupling force value. Default is None, depends on the model.
    :type min_force: int or float
    
    
    """

    def __init__(self, *args, **kwargs):

        self.name = kwargs.get('name')
        self.model = kwargs.get('model')
        self.variables = kwargs.get('variables')
        self.dynamic = kwargs.get('dynamic')
        
        self.max_force = None
        self.min_force = None

    def choose_model(self, neurons):

        """ 
        .. py::function:

        This function chooses the model according to the model name given.

        :param neurons: number of neurons in the system (samples)
        :type neurons: int
        
        """

        if self.name == 'Hodgkin-Huxley':
            
            self.model = {
                'HH':HodgkinHuxley(115,-12,86,120,36,.3,1),
                'Chemical':Chemical(0.1,-70),
                'SDE':SDE(0.02,0.5,1.0,115,-12,86,120,36,.3,1)
            }

            v = np.random.uniform(0.0,4.0,(neurons))
            m = self.model['HH'].m0(v)[2]
            n = self.model['HH'].n0(v)[2]
            h = self.model['HH'].h0(v)[2]
            y = self.model['Chemical'].y0(v)[2]
            I = np.zeros((neurons))+20

            self.variables = [v,m,n,h,y,I]

            self.dynamic = self.hh_dynamic

            self.max_force = 1.0
            self.min_force = 0.2

        elif self.name == 'Hindmarsh-Rose':
            
            self.model = HindmarshRose(1,3.0, 1.0,5.0,0.001,1.6,-1.6,1)

            x = np.random.uniform(-.5,.5,(neurons))
            y = np.random.uniform(-.5,.5,(neurons))
            z = np.random.uniform(-.5,.5,(neurons))
            sigma = np.zeros(shape=(neurons))+2.0

            self.variables = [x,y,z,sigma]

            self.dynamic = self.hr_dynamic

            self.max_force = 1.92
            self.min_force = 0.076

        elif self.name == 'Integrate-and-Fire':

            self.model = IntegrateAndFire(0,1.0,10,1.0)
            
            v = np.random.uniform(0,0.5,size=neurons)
            I = np.zeros(shape=(neurons))+2.5
            sigma = np.zeros(shape=(neurons))+0.2

            self.variables = [v,I,sigma]
            self.dynamic = self.iaf_dynamic

            
            self.max_force = 1.2
            self.min_force = 0.36

        elif self.name == 'GLM':

            self.model = GLM(
                neurons = neurons,
                spiketrain = np.ones(shape=(1,neurons)),
                response = 'delay',
                refractory = 'threshold',
                current = 'linear',
                threshold = -55
            )
            self.model.adjacency = np.ones(shape=(neurons,neurons))
            np.fill_diagonal(self.model.adjacency,0)

            self.model.weights = np.random.uniform(0,0.5,size=(neurons,neurons))
            self.model.step = 0.2
            self.model.window = 250
            self.model.time = 500

            delta, tau = .1,.1
            delay_response_args = [delta, tau]

            """ Refractory """
            amplitude, rest_pot, tau_2, refrac, hyper = 1,-70,0.1,0.01,100
            refractory_args = [amplitude, rest_pot, tau_2, refrac, hyper]

            self.model.theta_amp = 100
            self.model.delay = 0.3
            self.model.fire_beta = 0.05
            self.model.fire_tau = 10
            self.model.fire = 0.9

            i_ext = np.random.uniform(0,0.5,size=(neurons))
            w0 = np.random.uniform(0,1,size=(neurons))
            self.model.external_current(I_ext, w0)

            current_args = [i_ext,w0]

            self.variables = [delay_response_args, refractory_args, current_args]

            self.dynamic = self.glm_dynamic

            self.min_force = 0.468
            self.max_force = 0.72

        elif self.name == 'Rulkov':

            self.model = Rulkov(4.2,0.001,-1.2)

            x = np.random.uniform(-1,-1.2,size=neurons)
            y = np.zeros((neurons)) -2.9
            current = np.zeros((neurons))

            self.variables = [x,y,current]

            self.dynamic = self.rk_dynamic

            self.min_force = 0.01
            self.max_force = 0.22

        elif self.name == 'Izhikevic':

            self.model = Izhikevic(0.02,0.2,-55,0.93)

            v = np.random.uniform(-65,-64,size=neurons)
            u = 0.2*v
            I = np.zeros((neurons))+10

            self.variables = [v,u,I]

            self.dynamic = self.iz_dynamic

            self.min_force = 0.9
            self.max_force = 1.5

        elif self.name == 'CNV':

            self.model = CNV(0.002, 0.4,0.3,0.3,0.1,0.01)

            x = np.random.uniform(0,0.01,size=(neurons))
            y = np.zeros((neurons))
            j = np.zeros((neurons)) + 0.1123

            self.variables = [x,y,j]

            self.dynamic = self.cnv_dynamic

            self.min_force = 0.298
            self.max_force = 1.0

    def hh_dynamic(self,step,connections, *args):

        """ 
        .. py::function:

        This function computes the Hodgkin-Huxley Dynamic.

        :param step: time step.
        :type step: int or float
        :param connections: the coupling force square matrix
        :type connections: numpy.ndarray
        :param *args: list of variables of the biological neuron model
        :type *args: list
        
        """

        v,m,n,h,y,current = args[0][0], args[0][1], args[0][2], args[0][3], args[0][4], args[0][5]

        self.data = v
        coupling = self.synapse(connections)

        next_v = v + self.model['SDE'].membrane_potential(v,m,n,h,current) - self.model['Chemical'].synapse(y,v)*step - coupling
        next_m = m + self.model['SDE'].stochastic_sodium_activate(m,v)
        next_n = n + self.model['SDE'].stochastic_potassium_activate(n,v)
        next_h = h + self.model['SDE'].stochastic_sodium_deactivate(h,v)
        next_y = y + self.model['SDE'].stochastic_chemical_transmitter(y,v)

        return next_v,next_m,next_n,next_h,next_y,current

    def hr_dynamic(self,step,connections,*args):
        
        """ 
        .. py::function:

        This function computes the Hindmarsh-Rose Dynamic.

        :param step: time step.
        :type step: int or float
        :param connections: the coupling force square matrix
        :type connections: numpy.ndarray
        :param *args: list of variables of the biological neuron model
        :type *args: list
        
        """

        x,y,z,sigma = args[0][0], args[0][1], args[0][2], args[0][3]


        self.data = x
        coupling = self.synapse(self.connections)

        next_x = x + self.model.potential(x,y,z)*step + sigma*x*np.random.uniform(0,step,size=x.size) - coupling
        next_y = y + self.fast_ion_channel(x,y)*step + sigma*x*np.random.uniform(0,step,size=x.size)
        next_z = z + self.slow_ion_channel(x,z)*step + sigma*x*np.random.uniform(0,step,size=x.size)

        return next_x, next_y, next_z, sigma

    def iaf_dynamic(self, step,connections, *args):

        
        """ 
        .. py::function:

        This function computes the Integrate-and-Fire Dynamic.

        :param step: time step.
        :type step: int or float
        :param connections: the coupling force square matrix
        :type connections: numpy.ndarray
        :param *args: list of variables of the biological neuron model
        :type *args: list
        
        """

        v, I, sigma = args[0][0], args[0][1], args[0][2] 

        self.data = v
        coupling = self.synapse(connections)

        next_v = v + self.model.lif(v,I)*step + sigma*v*np.random.uniform(0,0.2,size=(v.size)) - coupling

        v = self.model.reset(v,next_v)

        return v, I, sigma

    def glm_dynamic(self,*args):
        
        """ 
        .. py::function:

        This function computes the Generalized Linear Model Dynamic.

        :param *args: list of variables of the biological neuron model
        :type *args: list
        
        """

        response_args, refractory_args, current_args = args[0], args[1], args[2]

        response_args.insert(0,self.model.last_spike)
        refractory_args.insert(0,self.model.last_spike)

        self.model.incoming_spikes(response_args)
        self.model.membrane_potential(refractory_args)
        self.model.generate_spikes()
        self.model.update()
        self.model.external_current(self.model.I,current_args[1], initializer=np.random.uniform(0,0.5))

    def rk_dynamic(self,step,connections, *args):

        
        """ 
        .. py::function:

        This function computes the Rulkov Dynamic.

        :param step: time step.
        :type step: int or float
        :param connections: the coupling force square matrix
        :type connections: numpy.ndarray
        :param *args: list of variables of the biological neuron model
        :type *args: list
        
        """

        x,y,current = args[0][0], args[0][1], args[0][2]

        self.data = x
        coupling = self.synapse(self.connections)

        x = self.model.fx(x,y,current) - coupling
        y = self.model.fy(x,y)

        return x,y,current

    def iz_dynamic(self,step, connections, *args):
        
        """ 
        .. py::function:

        This function computes the Izhikevic Dynamic.

        :param step: time step.
        :type step: int or float
        :param connections: the coupling force square matrix
        :type connections: numpy.ndarray
        :param *args: list of variables of the biological neuron model
        :type *args: list
        
        """

        v,u,current = args[0][0], args[0][1], args[0][2]

        self.data = v
        coupling = self.synapse(connections)

        next_v = v + self.model.potential(v,u,current)*step - coupling
        next_u = u + self.model.recovery(v,u)*step

        new = self.model.update(next_v,next_u)
        v,u = new[0],new[1]

        return v, u, current

    def cnv_dynamic(self,step,connections, *args):
        
        """ 
        .. py::function:

        This function computes the Courbage-Nekorkin-Vdovin Dynamic.

        :param step: time step.
        :type step: int or float
        :param connections: the coupling force square matrix
        :type connections: numpy.ndarray
        :param *args: list of variables of the biological neuron model
        :type *args: list
        
        """

        x,y,j = args[0][0], args[0][1], args[0][2]

        self.data = x
        coupling = self.synapse(connections)

        x = self.model.potential(x,y) - coupling
        y = self.model.recovery(x,y,j)

        return x,y,j


""" 
Classe Similaridade 
    função Inicializador
        Atributos:
            Escolhido
"""
class Similarity():

    """ 
    .. py::class:

    This class involkes all the similarities kernels
    
    """

    def __init__(self):
        super().__init__()
        self.chosen = {
            'Euclidean': euclidean,
            'Hamming': hamming,
            'Manhattan':manhattan,
            'Jaccard':jaccard,
            'Cosine':cosine,
            'IMED': imed
        }


""" 
Classe NSC
    função Inicializador
    função Calcula_Similaridade
    função Calcula_Peso
    função Inicializador de Pesos
    função Estabelece_Parâmetros
    função Obtém_Parâmetros
    função Busca_Conexões
    função Encontrar_Múltiplas_Incidências
    função Atualizar_Conexões
    função Atualizar_Modelo
    função Fit
"""
class NSC():

    """ 
    .. py::class:

    This class implements the Neuron Synchronization Competition algorithm.
    The semi-supervised method generates synchronized neurons based on the labeled data, and unsync for the
    unlabeled, then the method uses similarities between the samples to create coupling connections, and those couplings
    can be reinforced by the amount of connections inside a group, and punished by others. At the end, the labeled data are those
    that are synchronized and segmented in different times.
    
    .. py::function: __init__(self,*args,**kwargs)

    :param preprocessing: Instance of the Preprocessing class.
    :type preprocessing: Preprocessing
    :param similarity: Instance of the Similarity class
    :type similarity: Similarity
    :param neuron: Instance of the Neuron class
    :type neuron: Neuron
    :param alpha ($\alpha$): Decay parameter of the exponential term in the connection creation equation.
    :type alpha: int or float
    :param beta ($\beta$): Decay parameter of the punishment exponential term
    :type beta: into or float
    :param capacity: The vector of connections'number that each neuron can have
    :type capacity: numpy.ndarray
    :param categorical: The categorical data of the dataset,
    :type categorical: numpy.ndarray
    :param classified: The classified sample at time t. Default is None.
    :type classified: numpy.ndarray
    :param classified_number: The number of classified data at each iteration t
    :type classified_number: int
    :param confusion_matrix: The confusion matrix related to the classification of the NSC algorithm
    :type confusion_matrix: sklearn.metrics.confusion_matrix
    :param data: A DataFrame object of the dataset used for the implementation of the NSC algorithm
    :type data: pandas.core.frame.DataFrame
    :param degree_out: A matrix Neurons x Labels + 1 in which each element represents the out degree coming from label j to neuron i,
                        and the last column is the total incoming in unlabeled neuron i.
    :type degree_out: numpy.ndarray
    :param distance_name: The similarity name used to calculate the coupling force between the samples
    :type distance_name: str
    :param disputed: A vector that stores the indexes of the disputed neurons
    :type disputed: numpy.ndarray
    :param found: A dictionary that stores the indexes of the neurons found in the search phase for each label
    :type found: dict
    :param gamma: The decay parameter of the exponential term at the reinforce term.
    :type gamma: int or float 
    :param incident_degree: A matrix Neurons x Labels, where each element represents the amount of connections coming from the groups
                            different of j into the neuron i.
    :type incident_degree: numpy.ndarray
    :param indexes_of: A dictionary that stores the indexes of the labeled and unlabeled data.
    :type indexes-of: dict
    :param inner_degree: A dictionary that stores the amont of connections per label.
    :type inner_degree: dict
    :param labels_array: A matrix that stores the label that are passed from neuron i to neuron j
    :type labels_array: numpy.ndarray
    :param labels_list: The list of labels
    :type labels_list: list
    :param max_std: Maximum attribute standard deviation
    :type max_std: into or float
    :param neighbors: Maximum amount of connections that a neuron can make
    :type neighbors: int
    :param neuron.name: The name of the biological neuron model used.
    :type neuron.name: str
    :param neurons: The amount of neurons (samples)
    :type neurons: int
    :param numerical: The numerical dtype data in dataset
    :type numerical: numpy.ndarray
    :param print_steps: Boolean value that allows to print the steps of each phase in NSC algorithm
    :type print_steps: bool
    :param print_classification: Boolean value that allows to print the classification at each iteration
    :type print_classification: bool
    :param prints: A list that stores all the prints
    :type prints: list
    :param time_step: The time step for the neuron model
    :type time_step: int or float
    :param target: The column name that represents the target
    :type target: str
    :param threshold: The hypersphere radius
    :type threshold: int or float
    :param w_max: Maximum value of the coupling force
    :type w_min: int or float
    :param w_min: Minimum value of the coupling force
    :type w_min: int or float
    :param w_step: Coupling force step at reinforcement and punishment
    :type w_step: int or float
    :param weights: A matrix that stores all the coupling force between neurons i and j
    :type weights: numpy.ndarray
    :param X: Attributes data
    :type X: numpy.ndarray
    :param Y: Output data
    :type Y: numpy.ndarray
    :param y_predicted: Predicted output data
    :type y_predicted: numpy.ndarray
    """
    def __init__(self,*args,**kwargs):

        """ Instantiation """
        self.preprocessing = Preprocessing()
        self.similarity = Similarity()
        self.neuron = Neuron()

        """ Attributes """
        self.alpha = kwargs.get('alpha')
        self.beta = kwargs.get('beta')
        self.capacity = kwargs.get('capacity')
        self.categorical = kwargs.get('categorical')
        self.classified = None
        self.classified_number = 0
        self.confusion_matrix = None
        self.data = kwargs.get('data')
        self.degree_out = kwargs.get('degree_out')
        self.distance_name = kwargs.get('similarity')
        self.disputed = np.array([])
        self.found = {}
        self.gamma = kwargs.get('gamma')
        self.incident_degree = kwargs.get('incident')
        self.indexes_of = {}
        self.inner_degree = {}
        self.labels_array = kwargs.get('labels_array')
        self.labels_list = kwargs.get('labels')
        self.max_std = kwargs.get('max_std')
        self.neighbors = kwargs.get('neighbors')
        self.neuron.name = kwargs.get('model')
        self.neurons = kwargs.get('neurons')
        self.numerical = kwargs.get('numerical')
        self.print_steps = kwargs.get('print_steps')
        self.print_classification =kwargs.get('print_info')
        self.prints = []
        self.search_expand = kwargs.get('expand')
        self.time_step = kwargs.get('time_step')
        self.target= kwargs.get('target')
        self.threshold = kwargs.get('threshold')
        self.w_max = kwargs.get('w_max')
        self.w_min = kwargs.get('w_min')
        self.w_step = kwargs.get('w_step')
        self.weights = kwargs.get('weights')
        self.X = kwargs.get('X')
        self.Y = kwargs.get('Y')
        self.y_predicted = kwargs.get('y_predict')

        """ Code """
        if isinstance(self.distance_name,str):
            self.distance = self.similarity.chosen[self.distance_name]
        if (not isinstance(self.data,type(None))) and (isinstance(self.neuron.name,str)):
            self.neurons = self.data.shape[0]
            self.neuron.choose_model(self.neurons)
            self.w_max = self.neuron.max_force
            self.w_min = self.neuron.min_force
            self.y_predicted = -np.ones(shape=(self.neurons))
            self.degree_out = np.zeros(shape=(self.data.shape[0], np.unique(self.data['target']).size+1))
            self.capacity = np.zeros(shape=(self.neurons))
        if isinstance(self.labels_array,type(None)):
            self.labels_array = - np.ones(shape=(self.neurons,self.neurons)) 
        if isinstance(self.incident_degree,type(None)):
            self.incident_degree = np.zeros(shape=(self.data.shape[0],np.unique(self.data['target']).size))


    def calculate_similarity(self, x, y, axis=1):

        """ 
        .. py::function:

        This function calculates the similarity between the samples x and y
        
        :param x: Sample(s) x
        :type x: numpy.ndarray
        :param y: Sample(s) y
        :type y: numpy.ndarray
        :param axis: axis along theoperation will be realized
        :type axis: int

        """

        return self.distance(x,y,axis=axis)

    def create_connection(self,distance):

        """ 
        .. py::function:
        This function creates the connection between neurons based on their distances values

        :param distance: A matrix of distances between neurons i and j
        :type distance: numpy.ndarray
        
        """

        return (self.w_min + .5*(self.w_max-self.w_min))*np.exp(-self.alpha*distance)

    def initialize_weights(self, data_dtype):

        """ 
        ..py::function:

        This function initializes the weights, the inner degree and the labels array
        of the initial labeled data.

        :param data_dtype: Attributes data of a specific data type
        :type data_dtype: numpy.ndarray
        
        """
        
        self.weights = np.zeros(shape=(self.data.shape[0],self.data.shape[0]))

        for label in self.labels_list:
            labeled = self.indexes_of['label'][label]
            row_grid, col_grid = np.meshgrid(labeled, labeled.T)
            rows_total = row_grid.flatten()
            cols_total = col_grid.flatten()
            inds = np.where(rows_total==cols_total)[0]
            rows = np.delete(rows_total,inds)
            columns = np.delete(cols_total,inds)
            distances = self.calculate_similarity(
                data_dtype[labeled,None,:], data_dtype[labeled,:], axis=2
            )
            distances = distances.flatten()
            distances= np.delete(distances,inds)
            self.weights[rows,columns] = self.create_connection(distances)
            self.inner_degree[label] = int(len(self.indexes_of['label'][label])*(len(self.indexes_of['label'][label])-1))
            self.labels_array[rows,columns] = label
            self.labels_array[columns,rows] = label
            

    def set_parameters(self, **kwargs):

        """ 
        ..py::function:

        This function set the parameters (class attributes)
        
        """

        for key, value in kwargs.items:
            setattr(self,key,value)
        
        self.__init__()
 
    def get_parameters(self):

        """ 
        .. py::function:

        This function get the parameters (class attributes)
        
        """

        print(self.__dict__)

    def connection_search(self, unlabeled, data_dtype):

        """ 
        .. py::function:

        This function searchs for new connections

        :param unlabeled: unlabeled data point index
        :type unlabeled: int
        :param data_dtype: Attributes data of a specific data type
        :type data_dtype: numpy.ndarray
        
        """
        
        self.prints.append("\nSearch Phase ... \n")
        self.classified = None
        self.found = {label:np.array([]) for label in self.labels_list}
        for label in self.labels_list:
            similarities = self.calculate_similarity(
                data_dtype[self.indexes_of['label'][label],:],
                data_dtype[self.indexes_of['unlabel'][unlabeled],None,:]
            )
            self.neurons_found = np.where(similarities <= self.threshold)[0]
            if len(self.neurons_found) > 0:
                found_indexes = self.indexes_of['label'][label][self.neurons_found]
                below_capacity_indexes = found_indexes[np.where(self.capacity[found_indexes]<self.neighbors)[0]]
                self.neurons_below_capacity = below_capacity_indexes
                if self.neurons_below_capacity.size > 0:
                    self.classified = self.indexes_of['unlabel'][unlabeled]
                    self.weights[self.indexes_of['label'][label][self.neurons_found], self.indexes_of['unlabel'][unlabeled]] = self.create_connection(similarities[self.neurons_found])
                    self.labels_array[self.indexes_of['label'][label][self.neurons_found], self.indexes_of['unlabel'][unlabeled]] = label
                    self.degree_out[self.indexes_of['unlabel'][unlabeled],label] = self.neurons_found.size
                    self.degree_out[self.indexes_of['unlabel'][unlabeled],-1] += self.neurons_found.size
                    self.found[label] = self.indexes_of['label'][label][self.neurons_found]
                    self.inner_degree[label] += self.neurons_found.size
                    self.capacity[self.neurons_below_capacity] += 1
        if isinstance(self.classified,type(None)):
            self.prints.append("Nothing found!\n")
        else:
            self.prints.append("Found neuron {}\n".format(self.indexes_of['unlabel'][unlabeled]))

       

    def find_multiple_incidence(self,unlabeled):

        """ 
        .. py::function:

        This function searchs if there are multiple incidence in the unlabeled data sample.

        :param unlabeled: unlabeled data point index
        :type unlabeled: int
        
        """

        unlabeled = self.indexes_of['unlabel'][unlabeled]
        self.incoming_labels = self.labels_array[:,unlabeled]
        self.incoming = len(np.where(self.labels_array[:,unlabeled]!=-1)[0])
        self.uniques = np.unique(self.incoming_labels)
        self.size = len(self.uniques)

        if self.size > 2:
            self.prints.append("There are multiple incidence in {}\n".format(unlabeled))
            if unlabeled not in self.disputed:
                self.disputed = np.append(self.disputed,unlabeled)
            
            degree_out = self.degree_out[unlabeled,:]
            for label in self.labels_list:
                if degree_out[label] != 0:
                    self.incident_degree[unlabeled,label-1] = np.sum(degree_out[np.where(np.array(self.labels_list)!=label)])
            
        else:
            self.prints.append("There are no multiple incidence!\n")
            for label in self.labels_list:
                if self.found[label].size > 0:
                    self.indexes_of['label'][label] = np.setdiff1d(
                        np.unique(np.where(self.labels_array==label)[1]),self.disputed
                    )

                    self.weights[unlabeled,self.found[label]] = self.weights[self.found[label],unlabeled]
                    self.labels_array[unlabeled, self.found[label]] = label
            
    def cut_connections(self):

        """ 
        .. py::function:

        This function cuts the connections if there are coupling forces (weights) below the minimum coupling force (w_min)
        
        """

        self.prints.append("Cutting Conections Phase ... \n")
       
        self.rows_cuted, self.cols_cuted = np.where((self.weights!=0)&(self.weights<self.w_min))

        self.disputed_cols, index, count = np.unique(self.cols_cuted,return_index=True, return_counts=True)

        if self.disputed_cols.size > 0:
            self.prints.append("There are cuts\n")
            
            for i in range(self.disputed_cols.size):

                self.disconnected = self.rows_cuted[np.where(self.disputed_cols[i]==self.cols_cuted)]
                self.indexes = {}
                self.weights_means = {}
                self.indexes_size = np.zeros((len(self.labels_list)))

                for label in self.labels_list:
                    indexes = np.intersect1d(self.disconnected, np.where(self.labels_array[:,self.disputed_cols[i]]==label)[0])
                    self.indexes[label] = indexes
                    self.indexes_size[label-1] = indexes.size
                    if indexes.size > 0:
                        disputed = np.repeat(self.disputed_cols[i], indexes.size)
                        self.weights_means[label] = np.mean(self.weights[indexes,disputed])
                    else:
                        self.weights_means[label] = 0
                if self.disconnected.size == self.degree_out[self.disputed_cols[i],-1]:
                    self.prints.append("All connections have been severed\n")
        
                    label, max_weight = max(self.weights_means.items(), key=operator.itemgetter(1))
                    intersec = np.setdiff1d(self.disconnected, self.indexes[label])
                    self.capacity[intersec] -= 1
        
                    self.weights[intersec,np.repeat(self.disputed_cols[i],intersec.size)] = 0
                    self.weights[self.indexes[label],np.repeat(self.disputed_cols[i],self.indexes[label].size)] = self.w_min

                    self.labels_array[intersec, np.repeat(self.disputed_cols[i],intersec.size)] = -1

                    self.degree_out[self.disputed_cols[i],np.where(np.array(self.labels_list).astype(int)!=label)] = 0
                    self.degree_out[self.disputed_cols[i],-1] = self.degree_out[self.disputed_cols[i],label]

                    self.incident_degree[self.disputed_cols[i],:] = 0

                    for key,value in self.indexes.items():
                        if key!=label and value.size>0:
                            self.inner_degree[key] -= value.size
        
                    self.indexes_of['label'][label] = np.append(self.indexes_of['label'][label],self.disputed_cols[i])
                    self.disputed = np.setdiff1d(self.disputed, self.disputed_cols[i])

                elif self.disconnected.size < self.degree_out[self.disputed_cols[i],-1]:

                    self.prints.append("Only a few connections have been severed\n")
                    for label in self.labels_list:

                        if self.indexes[label].size > 0:

                            self.weights[self.indexes[label],np.repeat(self.disputed_cols[i],self.indexes[label].size)] = 0
                            self.capacity[self.indexes[label]] -= 1
                            self.labels_array[self.indexes[label], np.repeat(self.disputed_cols[i], self.indexes[label].size)] = -1
                            self.inner_degree[label] -= self.indexes[label].size
                            incident =0
                            if self.indexes_size[label-1] < self.degree_out[self.disputed_cols[i],label-1]:
                                for l in self.labels_list:
                                    if l != label:
                                        incident+=self.indexes[l].size

                            elif self.indexes_size[label-1]==self.degree_out[self.disputed_cols[i],label-1]:
                                incident = self.incident_degree[self.disputed_cols[i],label-1]
                            self.incident_degree[self.disputed_cols[i],label-1]-=incident

                            self.degree_out[self.disputed_cols[i],label]-=self.indexes[label].size
                            self.degree_out[self.disputed_cols[i],-1]-=self.indexes[label].size
                    unique_labels = np.unique(self.labels_array[:,self.disputed_cols[i]])
                    if unique_labels.size==2:
                        label = int(unique_labels[1])
                        self.indexes_of['label'][label] = np.append(self.indexes_of['label'][label],self.disputed_cols[i])
                        self.disputed = np.setdiff1d(self.disputed,self.disputed_cols[i])
                        self.incident_degree[self.disputed_cols[i],:] = 0
        else:
            self.prints.append("There are no cuts!")
        
    def update_weights(self):

        """ 
        .. py::function:

        This function updates the weights values with the reinforce and punishment terms and 
        then cut the connections by calling the cut_connections() method.
        
        """

        self.prints.append("Update Weights Phase...\n")
        for label in self.labels_list:
            self.row, self.col = np.where(self.labels_array==label)
            self.reinforce_exp = self.inner_degree[label]*np.exp(-self.gamma*self.incident_degree[self.col,label-1])

            self.punish_exp = self.incident_degree[self.col,label-1]*self.beta
            
            self.weights[self.row,self.col] = self.weights[self.row,self.col] + self.w_step*(1 - np.exp(-self.reinforce_exp)) \
                        - self.w_step*(1 - np.exp(-self.punish_exp))
            
            self.weights[self.row,self.col] = np.where(self.weights[self.row,self.col]>self.w_max,self.w_max,self.weights[self.row,self.col])
            
        self.cut_connections()
        
    def preprocess_data(self, shuffle = True, split = True, set_null = True, not_null = None, get_indexes = True, standarlize=True):
        
        """ 
        .. py::function:

        This function preprocess the dataset.

        :param shuffle: Boolean value that determines if will shuffle or not the data. Default is True.
        :type shuffle: bool
        :param split: Boolean value that determines if will split the data into features, numerical data, categorical data and output.
                      Default is True.
        :type split: bool
        :param set_null: Boolean value that determines if will set some output values to Null.
        :type set_null: bool
        :param not_null: Int, None or Dict values which determines how to choose the not null values. If int, then the program randomly
                         chooses this number of labeled data for each label, if None the program chooses randomly 10% of the data size,
                         if dict, for each label (key dict) the program choose the indexes determined in the dict (values dict).
        :type not_null: int, None or dict
        :param get_indexes: Boolean value that determines if will get or not the labeled and unlabeled data indexes 
        :type get_indexes: bool
        :param standarlize: Boolean value that determines if will standarlize the dataset
        :type standarlize: bool
        
        """


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
            elif isinstance(not_nul,type(None)):
                self.preprocessing.set_null_values(self.data, self.target, self.labels_list,label_size=int(0.1*self.data.shape[0]))
        if standarlize == True:
            if split == True:
                scaler = StandardScaler()

                self.X = scaler.fit_transform(self.X)
                self.numerical = scaler.fit_transform(self.numerical)

            elif split == False:

                features = self.data.drop(columns=[self.target],axis=1)

                numeric = features.select_dtypes(include=np.number)

                numeric_names = numeric.columns

                self.data.loc[:,numeric_names] = (numeric-numeric.mean())/numeric.std()


        if get_indexes == True: 

            self.indexes_of['unlabel'], self.indexes_of['label'] = self.preprocessing.get_label_unlabel_inds(self.data, self.target, self.labels_list)               
            self.indexes_of['unlabel'] = np.array(self.indexes_of['unlabel'])
        print("\n-------------------The data has been preprocessed --------------------\n")
    
    def fit(self, epochs, data_dtype):

        """ 
        .. py::function:

        This function fits the NSC algorithm at the dataset.

        :param epochs: Number of time, preferably a number higher than the number of neurons.
        :type epochs: int
        :param data_dtype: Attributes data of a specific data type
        :type data_dtype: numpy.ndarray
        
        """

        self.initialize_weights(data_dtype)

        self.neuron.potential = np.zeros(shape=(epochs,len(self.neuron.variables),self.neurons))

        i = 0
        j = 0
        
        num = pd.DataFrame(data_dtype)

        if self.distance_name=='Euclidean':
            self.max_std = 0
            for label in self.labels_list:
                self.classified_number = self.classified_number + self.indexes_of['label'][label].size
                std = num.loc[self.indexes_of['label'][label],:].std().max()
                if std > self.max_std:
                    self.max_std = std
            self.threshold = 6*self.max_std/self.search_expand
            while i < epochs or self.classified_number < self.neurons:
                
                self.prints = []
                self.classified_number = 0

                if j >= self.indexes_of['unlabel'].size:
                    self.prints.append("It has reached the end of the Unlabeled array, returning to index 0 \n")
                    j = 0
                if self.indexes_of['unlabel'].size>0:
                    self.connection_search(j,data_dtype) # OKAY
                    self.find_multiple_incidence(j)

                self.update_weights()
                #self.neuron.variables = self.neuron.dynamic(self.time_step, self.weights, self.neuron.variables)
                #self.neuron.potential[i] = np.array([var for var in list(self.neuron.variables)])
                
                if not isinstance(self.classified,type(None)):
                    self.indexes_of['unlabel'] = np.setdiff1d(self.indexes_of['unlabel'],self.classified)
                else:
                    j = j+1
                i = i+1
                if i == epochs:
                    break
                if self.indexes_of['unlabel'].size !=0:
                    self.threshold = self.threshold + 6*self.max_std/self.search_expand
                for label in self.labels_list:
                    self.classified_number = self.classified_number + self.indexes_of['label'][label].size
                
                
                if self.print_steps == True:
                    for k in range(len(self.prints)):
                        print(self.prints[k])
                if self.print_classification == True:
                    print("\nIteration: ", i)
                    print("Disputed Neurons: ", self.disputed)
                    print("Classified: ", self.classified_number)
                    print("Unlabeled: ", self.indexes_of['unlabel'].size)
            for label in self.labels_list:
                self.y_predicted[self.indexes_of['label'][label]] = label
            self.confusion_matrix= confusion_matrix(self.Y, self.y_predicted)
        
        # elif self.distance_name=='Hamming':
        #     self.max_std = self.neurons*0.5/6
        #     self.threshold = 1/self.neurons

        #     while i < epochs or self.classified_number < self.neurons:
                
        #         self.prints = []
        #         self.classified_number = 0

        #         if j >= self.indexes_of['unlabel'].size:
        #             self.prints.append("It has reached the end of the Unlabeled array, returning to index 0 \n")
        #             j = 0
        #         if self.indexes_of['unlabel'].size>0:
        #             self.connection_search(j,data_dtype) # OKAY
        #             self.find_multiple_incidence(j)

        #         self.update_weights()
        #         #print('Max: ', self.weights.max(), ', Min: ', self.weights.min())
        #         # self.neuron.variables = self.neuron.dynamic(self.time_step, self.weights, self.neuron.variables)
        #         # self.neuron.potential[i] = np.array([var for var in list(self.neuron.variables)])
                
        #         if not isinstance(self.classified,type(None)):
        #             self.indexes_of['unlabel'] = np.setdiff1d(self.indexes_of['unlabel'],self.classified)
        #         else:
        #             j = j+1
        #         i = i+1
        #         if i == epochs:
        #             break
        #         if self.indexes_of['unlabel'].size !=0:
        #             self.threshold = self.threshold + 6*self.max_std/self.search_expand
        #         for label in self.labels_list:
        #             self.classified_number = self.classified_number + self.indexes_of['label'][label].size
        #         if self.classified_number == self.neurons:
        #             break
        #             # std = num.loc[self.indexes_of['label'][label],:].std().max()
        #             # if std > self.max_std:
        #             #     self.max_std = std

                
        #         # if self.print_steps == True:
        #         #     for k in range(len(self.prints)):
        #         #         print(self.prints[k])
        #         # if self.print_classification == True:
        #         #     print("\nIteration: ", i)
        #         #     print("Disputed Neurons: ", self.disputed)
        #         #     print("Classified: ", self.classified_number)
        #         #     print("Unlabeled: ", self.indexes_of['unlabel'].size)
        #     for label in self.labels_list:
        #         self.y_predicted[self.indexes_of['label'][label]] = label
        #     self.confusion_matrix= confusion_matrix(self.Y, self.y_predicted)

    def predict(self, input_array, data_dtype):

        """ 
        .. py::function:

        This function predicts the labels of the input_array

        :param input_array: The feature data to predict their classes.
        :type input_array: numpy.ndarray
        :param data_dtype: Attributes data of a specific data type
        :type data_dtype: numpy.ndarray
        
        """

        data_dtype = data_dtype.copy()
        data_dtype = np.vstack((data_dtype,input_array))

        prediction = -np.ones(input_array.shape[0])

        labeled_by = self.indexes_of['label']

        weights = np.zeros(shape=(self.neurons+input_array.shape[0],self.neurons+input_array.shape[0]),dtype=self.weights.dtype)
        weights[:self.weights.shape[0],:self.weights.shape[1]] = self.weights
        
        labels_array = - np.ones(shape=(self.neurons+input_array.shape[0],self.neurons+input_array.shape[0]),dtype=self.labels_array.dtype)
        labels_array[:self.labels_array.shape[0],:self.labels_array.shape[1]] = self.labels_array

        inner_degree = self.inner_degree
        incident_degree = np.vstack((self.incident_degree,np.zeros(shape=(input_array.shape[0],self.incident_degree.shape[1]), dtype=self.incident_degree.dtype)))
        
        degree_out = np.vstack((self.degree_out,np.zeros(shape=(input_array.shape[0], self.degree_out.shape[1]),dtype=self.degree_out.dtype)))
        
        
        shift = input_array.shape[0]

        disputed = np.array([])

        for i in range(len(input_array)):
            print(i)

            """ SEARCH AND CONNECTIONS """
            found = {label:np.array([]) for label in self.labels_list}
            for label in self.labels_list:

                similarities = self.calculate_similarity(
                data_dtype[labeled_by[label],:],
                input_array[i,None,:]
                )

                neurons_found = np.where(similarities<= self.threshold)[0]

                if len(neurons_found) > 0:
                    
                    found_indexes = labeled_by[label][neurons_found]

                    classified = input_array[i]
                    weights[labeled_by[label][neurons_found], i+self.neurons] = self.create_connection(similarities[neurons_found])
                    labels_array[labeled_by[label][neurons_found],i+self.neurons] = label
                    degree_out[i+self.neurons,label] = neurons_found.size
                    degree_out[i+self.neurons,-1] += neurons_found.size
                    found[label] = labeled_by[label][neurons_found]
                    inner_degree[label] += neurons_found.size


            """ FIND MULTIPLE INCIDENCE """
            incoming_labels = labels_array[:,i+self.neurons]

            incoming = len(np.where(labels_array[:,i+self.neurons]!=-1)[0])
            uniques = np.unique(incoming_labels)
            size = len(uniques)

            if size > 2:
               
                if i+self.neurons not in disputed:
                    disputed = np.append(disputed,i+self.neurons)
                
                degree_array_out = degree_out[i+self.neurons,:]
                for label in self.labels_list:
                    if degree_array_out[label] != 0:
                        incident_degree[i+self.neurons,label] = np.sum(degree_array_out[np.where(np.array(self.labels_list)!=label)])
                
            else:
                for label in self.labels_list:
                    if found[label].size > 0:
                        labeled_by[label] = np.setdiff1d(
                            np.unique(np.where(labels_array==label)[1]),disputed
                        )

                        weights[i+self.neurons,found[label]] = weights[found[label],i+self.neurons]
                        labels_array[i+self.neurons, found[label]] = label
                        
                        
            for label in self.labels_list:
                row, col = np.where(labels_array==label)
                reinforce_exp = inner_degree[label]*np.exp(-self.gamma*incident_degree[col,label])

                punish_exp = incident_degree[col,label]*self.beta
                
                weights[row,col] = weights[row,col] + self.w_step*(1 - np.exp(-reinforce_exp)) \
                            - self.w_step*(1 - np.exp(-punish_exp))
                
                weights[row,col] = np.where(weights[row,col]>self.w_max,self.w_max,weights[row,col])
                
            
            rows_cuted, cols_cuted = np.where((weights!=0)&(weights<self.w_min))

            disputed_cols, index, count = np.unique(cols_cuted,return_index=True, return_counts=True)

            if disputed_cols.size > 0:
                for j in range(disputed_cols.size):

                    disconnected = rows_cuted[np.where(disputed_cols[j]==cols_cuted)]
                    indexes = {}
                    weights_means = {}
                    indexes_size = np.zeros((len(self.labels_list)))

                    for label in self.labels_list:
                        inds = np.intersect1d(disconnected, np.where(labels_array[:,disputed_cols[j]]==label)[0])
                        indexes[label] = inds
                        indexes_size[label] = inds.size
                        if inds.size > 0:
                            disput = np.repeat(disputed_cols[j], inds.size)
                            weights_means[label] = np.mean(weights[inds,disput])
                        else:
                            weights_means[label] = 0
                    if disconnected.size == degree_out[disputed_cols[j],-1]:
                       
                        label, max_weight = max(weights_means.items(), key=operator.itemgetter(1))
                        intersec = np.setdiff1d(disconnected, indexes[label])
                        

                        weights[intersec,np.repeat(disputed_cols[j],intersec.size)] = 0
                        weights[indexes[label],np.repeat(disputed_cols[j],indexes[label].size)] = self.w_min

                        labels_array[intersec, np.repeat(disputed_cols[j],intersec.size)] = -1

                        degree_out[disputed_cols[j],np.where(np.array(self.labels_list).astype(int)!=label)] = 0
                        degree_out[disputed_cols[j],-1] = degree_out[disputed_cols[j],label]

                        incident_degree[disputed_cols[j],:] = 0

                        for key,value in indexes.items():
                            if key!=label and value.size>0:
                                inner_degree[key] -= value.size

                        labeled_by[label] = np.append(labeled_by[label],disputed_cols[j])
                        disputed = np.setdiff1d(disputed, disputed_cols[j])

                    elif disconnected.size < degree_out[disputed_cols[j],-1]:
                        for label in self.labels_list:

                            if indexes[label].size > 0:

                                weights[indexes[label],np.repeat(disputed_cols[j],indexes[label].size)] = 0
                                labels_array[indexes[label], np.repeat(disputed_cols[j], indexes[label].size)] = -1
                                inner_degree[label] -= indexes[label].size
                                incident =0
                                if indexes_size[label] < degree_out[disputed_cols[j],label]:
                                    for l in self.labels_list:
                                        if l != label:
                                            incident+=indexes[l].size

                                elif indexes_size[label]==degree_out[disputed_cols[j],label]:
                                    incident = incident_degree[disputed_cols[j],label]
                                incident_degree[disputed_cols[j],label]-=incident

                                degree_out[disputed_cols[j],label]-=indexes[label].size
                                degree_out[disputed_cols[j],-1]-=indexes[label].size
                        unique_labels = np.unique(labels_array[:,disputed_cols[j]])
                        if unique_labels.size==2:
                            label = int(unique_labels[1])
                            labeled_by[label] = np.append(labeled_by[label],disputed_cols[j])
                            disputed = np.setdiff1d(disputed,disputed_cols[j])
                            incident_degree[disputed_cols[j],:] = 0
        
        for label in self.labels_list:

            labels = labeled_by[label][np.where(labeled_by[label]>=self.neurons)[0]]
            prediction[labels.astype(int)-self.neurons] = label

        return prediction
