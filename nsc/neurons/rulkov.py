
'''
Author: Guilherme Marino Toso
Title = rulkov
Project = Semi-Supervised Learning Using Competition for Neurons' Synchronization
Package = neurons

Discription:
   A Neural Dynamical System is a set of differential equations that discribes
   the time evolution of the differential equations' variables as the action
   potential generated by a neuron and which is a variable in common with all
   the models. The model presented here is the Rulkov. This model 
   was first studied at a graduation project called "Análise de Sincronização em 
   Modelos de Osciladores Acoplados". 

'''
import numpy as np


class Rulkov():
    
    '''
    
    This is a Rulkov Neural Dynamical System Model
    
    '''
    
    
    def __init__(self, a, mu, sigma):
        
        '''
        
        Args:
            a: nonlinearity parameter
            mu: control parameter
            sigma: electrical input
        
        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of parameters values, means there's a 1-dimensional
            array of neurons.

            Note: the parameters must have the same shape.
        
        Data Type:

            It can be integer ('int') or floating ('float') values.

        Returns the paameters of the Rulkov Model.
        '''
        
        self.a = a
        self.mu = mu
        self.sigma = sigma
        
        
        
    def fx(self, x, y, current):
        
        '''
        
        This method calculates the fast variable.
        Although this variable has no phenomenological meaning it can be
        interpreted as the membrane potential.

        Args:

            x: fast variable, interpreted as a membrane potential
            y: slow variable, interpreted as a recovery variable
            current: electrical input.



        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of parameters values, means there's a 1-dimensional
            array of neurons.

            Note: If there's more than one value in the parameters array,
                  necessarily must have the same shape as the variables x and y.
        
        Data Type:

            It can be integer ('int') or floating ('float') values.

        Returns the new value of the fast variable.
        '''
        
        self.function = self.a/(1+x**2) + y + current
        
        return self.function
    
    def fy(self,x,y):
        
        '''
        
        This method calculates the slow variable.
        This property is due to the u parameter
        
        Args:

            x: fast variable, interpreted as a membrane potential
            y: slow variable, interpreted as a recovery variable
         

        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of parameters values, means there's a 1-dimensional
            array of neurons.

            Note: If there's more than one value in the parameters array,
                  necessarily must have the same shape as the variables x and y.
        
        Data Type:

            It can be integer ('int') or floating ('float') values.

        Returns the new value of the slow variable.
        
        '''
        
        self.yps = y - self.mu*(x - self.sigma)
        
        return self.yps
    
    def update(self,x,y,current):
        
        '''
        
        This method calculates the fast and slow variables and updates them
        
        Returns the new values of the fast and slow variables.

        '''
        
        self.dfast = self.fx(x,y,current)
        self.dslow = self.fy(x,y)
        
        x = self.dfast
        y = self.dslow


    
        return x, y