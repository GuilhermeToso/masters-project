'''
Author: Guilherme Marino Toso
Title = cnv
Project = Semi-Supervised Learning Using Competition for Neurons' Synchronization
Package = neurons

Discription:
   A Neural Dynamical System is a set of differential equations that discribes
   the time evolution of the differential equations' variables as the action
   potential generated by a neuron and which is a variable in common with all
   the models. The model presented here is the Courbage-Nekorkin-Vdovin (CNV).
   This model were first studied at a graduation project called "Análise de Sincroni-
   zação em Modelos de Osciladores Acoplados". 

'''
import numpy as np
    
class CNV():
    
    '''
    
    This class claculates the Courbage-Nekorkin-Vdovin (CNV) Neural Dynamical
    System Model.
    
    '''
    
    def __init__(self, a, m0, m1, d, beta, e):
        
        
        '''
        
        Args:
            a: Control parameter
            m0: Positive constants
            m1: Positive constants
            d (d > 0): Parameter that controls the threshold property of bursting 
                    oscillations
            beta (beta > 0): Same as d
            e (e > 0): Time scale of the recovery variable

        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of parameters values, means there's a 1-dimensional
            array of neurons.

            Note: the parameters must have the same shape.
        
        Data Type:

            It can be integer ('int') or floating ('float') values.       
        
        Returns the parameters of the Courbage-Nekorkin-Vdovin Model

        '''
        
        self.a = a
        self.m0 = m0
        self.m1 = m1
        self.d = d
        self.beta = beta
        self.e = e
        
        
    def heavside(self,x):
    
        '''
        
        This function calculates the value, array or matrice of diference
        between the potential and the d value, and returns a new data structure
        (same as x) based on heavside concept. If any value in x-d is higher 
        than 0, than in the new data structure it receives 1, else, receives 0

        Args:

            x: membrane potential

        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of x and y values, means there's a 1-dimensional
            array of neurons.

            Note: the parameters must have the same shape as x if there
                  is more than one value in the array of parameters.
        
        Data Type:

            It can be integer ('int') or floating ('float') values.
        
        Returns an array of 0's (if x-d < 0) and 1's (if x-d >= 0)

        '''
        
        val = x - self.d
        
        h = np.where(val>=0,1,0)
        
        return h
    
    
    
    def stimulus(self):
        
        '''
        
        This function calculates the max and min currents

        
        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of x and y values, means there's a 1-dimensional
            array of neurons.

            Note: the parameters must have the same shape as the amount of neurons
        Data Type:

            It can be integer ('int') or floating ('float') values.
        
        Returns the minimum and maximum currents.

        '''
        
        jmin= self.a*self.m1/(self.m0 + self.m1)
        
        jmax = (self.m0 + self.a*self.m1)/(self.m0 + self.m1)
        
        return jmin, jmax
        
    
    def function(self,x):
        
        
        '''
        
        This function calculates the linearity relation of the potential 
        variable
        
        Args:

            x: membrane potential        
        
        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of x and y values, means there's a 1-dimensional
            array of neurons.

            Note: the parameters must have the same shape as x if there
                  is more than one value in the array of parameters.
        
        Data Type:

            It can be integer ('int') or floating ('float') values.
        
        Returns the linearity relation of the membrane potential
        '''
        
        self.current = self.stimulus()
        jmin = self.current[0]
        jmax = self.current[1]
        
        func = np.where(x<=jmin,-self.m0*x,x)
        func1 = np.where((jmin<x)&(x<jmax),self.m1*(x-self.a),func)
        func2 = np.where(x>=jmax, -self.m0*(x-1), func1)
        
        
        return func2
    
    
    
    def potential(self,x,y):
        
        '''
        
        This function calculates the potential variable

        Args:

            x: membrane potential
            y: recovery variable

        
        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of x and y values, means there's a 1-dimensional
            array of neurons.

            Note: the parameters must have the same shape as x and y if there
                  is more than one value in the array of parameters.
        
        Data Type:

            It can be integer ('int') or floating ('float') values.
        
        Returns the new value of the membrane potential
        
        '''
        
        x_val = x + self.function(x) - y - self.beta*self.heavside(x)
        
        return x_val
    
    
    def recovery(self,x,y,j):
        
        '''
        
        This function calculates the recovery variable
        and j represents a external stimulus.

        Args:

            x: membrane potential
            y: recovery variable
            j: external stimulus
        
        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of x and y values, means there's a 1-dimensional
            array of neurons.

            Note: the parameters must have the same shape as x and y if there
                  is more than one value in the array of parameters.
        
        Data Type:

            It can be integer ('int') or floating ('float') values.
        
        Returns the new value of the recovery variable.
        
        '''
        
        y_val = y + self.e*(x - j)
        
        return y_val