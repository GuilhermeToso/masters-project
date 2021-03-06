# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 22:41:45 2019

Author: Guilherme Marino Toso
Title = hodgkinhuxley
Project = Semi-Supervised Learning Using Competition for Neurons' Synchronization
Package = neurons

Discription:
   A Neural Dynamical System is a set of differential equations that discribes
   the time evolution of the differential equations' variables as the action
   potential generated by a neuron and which is a variable in common with all
   the models. The model presented here is the Hodgkin - Huxley based on their
   work at 1952. This model was first studied at a graduation project called 
   "Análise de Sincronização em Modelos de Osciladores Acoplados". 

"""

import numpy as np


class HodgkinHuxley():

    """ 

    Hodgkin and Huxley Biological Neuron Model Class

    This class models the dynamical system of four first order ODEs
    developed by Hodgkin and Huxley in 1952.

    It calculates the proportion of the activating sodium molecules (SodiumActivation, or m),
    the proportion of the deactivating sodium molecules (SodiumDeactivation, or h), the proportion
    of the potassium activation molecules (PotassiumDeactivation, or n), the membrane potential 
    (Potential, or v) and finally, the initial proportions (m0, n0 and h0).

    See: Hodgkin, A. L. and Huxley, A. F. (1952). A quantitative description of membrane current
         and its application to conduction and excitation in nerve. The Journal of physiology,
         117(4):500–544.

    """
    
    
    def __init__(self, vna, vk, vl, gna, gk, gl, C):
        

        """ 
        Args:
            vna : Equilibrium potential for the sodium (Na) ions.
            vk  : Equilibrium potential for the potassium (K) ions.
            vl  : Equilibrium potential at which the 'leakage current' is zero.
            gna : Ionic conductance of the Sodium (Na).
            gk  : Ionic conductance of the Potassium (K).
            gl  : Ionic conductance of the ions from the leak current.   
            C   : Capacitance of the membrane cell.

        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of parameters values, means there's a 1-dimensional
            array of neurons.

            Note: the parameters must have the same shape.
        
        Data Type:

            It can be integer ('int') or floating ('float') values.

        Returns the parameters of the Hodgkin-Huxley Model

        """

        
        self.vna = vna
        self.vk = vk
        self.vl = vl
        self.gna = gna
        self.gk = gk
        self.gl = gl
        self.C = C
    

    def sodium_activation(self,m,v):

        """  
        
        SodiumActivation, or the 'm' value, represents the proportion of activating
        sodium molecules.
        
        Args:

            v: membrane potential
            m: proportion of activating molecules  

        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of potential (v), and m values, means there's a 
            1-dimensional array of neurons.

            Note: v and m must have same shapes.
        
        Data Type:

            It can be integer ('int') or floating ('float') values.

        Returns the value of the proportion of the activating molecules (m)

        """

        alpha_m = 0.1*((25.0-v)/(np.exp((25.0-v)/10.0)-1))
        
        beta_m = 4.0*np.exp(-v/18.0)
        
        # alpha_m = 0.1*(v+40)/(1 - np.exp(-(v+40)/10))
        # beta_m = 4*np.exp(-(v+65)/18)

        m_value = alpha_m*(1.0 - m) - beta_m*m

        return alpha_m, beta_m, m_value


    def sodium_deactivation(self,h,v):

        """  
        
        h value represents the proportion of inactivating molecules.

        Args:

            v: membrane potential
            h: proportion of inactivating molecules

        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of potential (v), and h values, means there's a 
            1-dimensional array of neurons.

            Note: v and h must have same shapes.
        
        Data Type:

            It can be integer ('int') or floating ('float') values.
        
        Returns the proportion of inactivating molecules (h)

        """

        alpha_h = 0.07*np.exp(-v/20.0)

        beta_h = 1/(1+np.exp((30.0-v)/10.0))

        # alpha_h = 0.07*np.exp(-(v+65)/20)
        # beta_h = 1/(1+np.exp(-(v+35)/10))

        h_value = alpha_h*(1.0 - h) - beta_h*h

        return alpha_h, beta_h, h_value

    
    def potassium_activation(self,n,v):

        """  
        
        n value represents the proportion of ionic particles.

        Args:

            v: membrane potential
            n: proportion of ionic particles

        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of potential (v), and n values, means there's a 
            1-dimensional array of neurons.

            Note: v and n must have same shapes.
        
        Data Type:

            It can be integer ('int') or floating ('float') values.
        
        Returns the proportion of ionic particles (n)
        
        """
        alpha_n = 0.01*((10.0-v)/(np.exp((10.0-v)/10.0)-1))
       
        beta_n = 0.125*np.exp(-v/80.0)

        # alpha_n = 0.01*(v+55)/(1 - np.exp(-(v+55)/10))
        # beta_n = 0.125*np.exp(-(v+65)/80)

        n_value = alpha_n*(1.0 - n) - beta_n*n

        return alpha_n, beta_n, n_value


    def potential(self,v,m,n,h,current):

        """  
        
        volts: The membrane potential that depends on an external current
        (I), and the sodium, potassium and leak ionic currents (Na, K, l).

        Args:

            v: membrane potential
            h: proportion of inactivating molecules
            m: proportion of activating molecules
            n: proportion of ionic molecules
            I: external electrical current

        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of potential (v) and rates of proportions (m, n and h),
            means there's a 1-dimensional array of neurons.

            Note: The parameters at __init__ not necesserily must have
            n-dimensions (potential v dimension at maximum), it can be 
            the same value for all neurons.

        Data Type:

            It can be integer ('int') or floating ('float') values.

        Returns the membrane potential

        """
        i_na = self.gna*(m**3.0)*h*(v - self.vna)

        i_k = self.gk*(n**4.0)*(v - self.vk)

        i_l = self.gl*(v - self.vl)

        volts = (current - i_na - i_k - i_l)/self.C
        return volts

    def m0(self,v):

        """  
        
        m0 : The initial value of the proportion of activating molecules.

        Args:

            v: initial membrane potential

        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of potential (v), means there's a 
            1-dimensional array of neurons.

        Data Type:

            It can be integer ('int') or floating ('float') values.

        Returns the initial value of the proportion of activating molecules

        """

        alpha_m = 0.1*((25.0-v)/(np.exp((25.0-v)/10.0)-1))
        
        beta_m = 4.0*np.exp(-v/18.0)

        # alpha_m = 0.1*(v+40)/(1 - np.exp(-(v+40)/10))
        # beta_m = 4*np.exp(-(v+65)/18)

        m_zero = alpha_m/(alpha_m + beta_m)
        
        return alpha_m, beta_m, m_zero

    def n0(self,v):

        """  
        
        n0 : The initial value of the proportion of ionic molecules.
        
        Args:

            v: initial membrane potential
        
        
        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of potential (v), means there's a 
            1-dimensional array of neurons.

        Data Type:

            It can be integer ('int') or floating ('float') values.
        
        Returns the initial value of the proportion of ionic molecules

        """
        alpha_n = 0.01*((10.0-v)/(np.exp((10.0-v)/10.0)-1))
       
        beta_n = 0.125*np.exp(-v/80.0)

        # alpha_n = 0.01*(v+55)/(1 - np.exp(-(v+55)/10))
        # beta_n = 0.125*np.exp(-(v+65)/80)
        
        n_zero = alpha_n/(alpha_n + beta_n)
        
        return alpha_n, beta_n, n_zero

    def h0(self,v):

        
        """  
        
        h0 : The initial value of the proportion of inactivating molecules.

        Args:

            v: initial membrane potential

        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of potential (v), means there's a 
            1-dimensional array of neurons.

        Data Type:

            It can be integer ('int') or floating ('float') values.

        Returns the initial value of the proportion of inactivating molecules


        """

        alpha_h = 0.07*np.exp(-v/20.0)

        beta_h = 1/(1+np.exp((30.0-v)/10.0))

        # alpha_h = 0.07*np.exp(-(v+65)/20)
        # beta_h = 1/(1+np.exp(-(v+35)/10))

        h_zero = alpha_h/(alpha_h + beta_h)
        
        return alpha_h, beta_h, h_zero



class Chemical():

    """ 
    
    Chemical Neurotransmitter Molecules' Proportion Class

    This class calculates the initial and voltage-dependent value of the y variable
    representing the proportion of chemical neurotransmitters.
    
    See: Baladron, J., Fasoli, D., Faugeras, O., & Touboul, J. (2012). Mean-field description
         and propagation of chaos in networks of Hodgkin-Huxley and FitzHugh-Nagumo neurons. 
         The Journal of Mathematical Neuroscience, 2(1), 10.

    """

    def __init__(self, ChemicalConductance, ReversalPotential):

        """ 
        
        The constructor method

        Args:

            ChemicalConductance: the chemical conductance for the chemical synapse
            ReversalPotential: the reverse membrane potential of the chemical term
        
        """

        self.jch = ChemicalConductance
        self.v_rev = ReversalPotential


    def channel(self, y, v):


        """  
        
        y value represents the proportion of Chemical Neurontransmitter molecules.

        Args:

            v: membrane potential
            y: proportion of chemical neurontransmitter molecules

        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of potential (v), and h values, means there's a 
            1-dimensional array of neurons.

            Note: v and y must have same shapes.
        
        Data Type:

            It can be integer ('int') or floating ('float') values.
        
        Returns the proportion of chemical neurontransmitter molecules (y)

        """


        alpha = 5./(1 + np.exp(-0.2*(v - 2.)))
        
        beta = 0.18

        #chemical_channel = alpha*(1 - y) + beta*y
        chemical_channel = alpha*(1 - y) - beta*y

        return alpha, beta, chemical_channel


    def y0(self,v):

        
        """  
        
        y0 : The initial value of the chemical neurotransmitter proportion .

        Args:

            v: initial membrane potential

        Dimensions:

            It can be a n-dimensional Numpy array. Each value is 
            respective to a neuron, so, for example: A 1-dimensional
            array of potential (v), means there's a 
            1-dimensional array of neurons.

        Data Type:

            It can be integer ('int') or floating ('float') values.

        Returns the initial value of chemical neurotransmitter proportion

        """
        alpha_y = 5./(1 + np.exp(-0.2*(v - 2.)))
        
        beta_y = 0.18

        y_zero = alpha_y/(alpha_y + beta_y)
        
        return alpha_y, beta_y, y_zero


    def synapse(self, y, v):

        """ 
        
        The chemical synapse term.
        
        Args:

            y: chemical neurontransmitter transfer rate
            v: membrane potential
        
        """

        return np.sum(self.jch*y*(v - self.v_rev))/v.size


class SDE(HodgkinHuxley):

    """ 
    
    The Stochastic Differential Equations of the Hodgkin-Huxley Model Class.

    This class takes the Hodgkin-Huxley dynamical system equations (V, M, N and H) also the chemical 
    neurotransmitters proportion (Y) and transform them into a Ornstein-Uhlenbeck equation (stochastic differential equations). 


    """

    def __init__(self, time_step, sigma_external, sigma, vna, vk, vl, gna, gk, gl, C):

        """ 
        
        The constructor method.
        
        Args:
            Hodgkin-Huxley Parameters: vna, vk, vl, gna, gk, gl and C.
            timeStep: time step for the Euler-Maruyama Method
            sigma_external: the stochastic voltage term strengh
            sigma: the stochastic term strengh  
        
        """
        
        HodgkinHuxley.__init__(self, vna, vk, vl, gna, gk, gl, C)
        self.dt = time_step
        self.sigma = sigma
        self.sigma_external = sigma_external


    def stochastic_sodium_activate(self, x, v):

        """ 
        
        This method transforms the Sodium activation gates function into
        stochastic functions

        Args:

            alpha: ion transfer rate
            beta: ion transfer rate
            x: dimenssionless variable that represents the sodium ion's proportions (m) 
        
        """
        # alpha = 0.1*(v+40)/(1 - np.exp(-(v+40)/10))
        # beta = 4*np.exp(-(v+65)/18)

        alpha = 0.1*((25.0-v)/(np.exp((25.0-v)/10.0)-1))
        
        beta = 4.0*np.exp(-v/18.0)
        
        
        dw = np.random.normal(0, np.sqrt(self.dt), size = x.size)

        zeros = np.zeros(x.size)

        langevin = np.where((x>0)&(x<1),np.sqrt(alpha*(1 - x) + beta*x)*0.1*np.exp(-0.5/(1 - (2*x - 1)**2)), zeros)

        return (alpha*(1.0 - x) - beta*x)*self.dt + self.sigma*langevin*dw

    def stochastic_sodium_deactivate(self, x, v):

        """ 
        
        This method transforms the Sodium ion deactivation gates function into
        stochastic functions

        Args:

            alpha: ion transfer rate
            beta: ion transfer rate
            x: dimenssionless variable that represents the Sodium ion's proportions (h) 
        
        """

        # alpha = 0.07*np.exp(-(v+65)/20)
        # beta = 1/(1+np.exp(-(v+35)/10))

        alpha = 0.07*np.exp(-v/20.0)

        beta = 1/(1+np.exp((30.0-v)/10.0))
        
        
        dw = np.random.normal(0, np.sqrt(self.dt), size = x.size)

        zeros = np.zeros(x.size)

        langevin = np.where((x>0)&(x<1),np.sqrt(alpha*(1 - x) + beta*x)
            *0.1*np.exp(-0.5/(1 - (2*x - 1)**2)), zeros)


        return (alpha*(1.0 - x) - beta*x)*self.dt + self.sigma*langevin*dw


    def stochastic_potassium_activate(self, x, v):

        """ 
        
        This method transforms the Potassium ion activation gates function into
        stochastic functions

        Args:

            alpha: ion transfer rate
            beta: ion transfer rate
            x: dimenssionless variable that represents the Potassium ion's proportions (n) 
        
        """
        alpha = 0.01*((10.0-v)/(np.exp((10.0-v)/10.0)-1))
       
        beta = 0.125*np.exp(-v/80.0)

        # alpha = 0.01*(v+55)/(1 - np.exp(-(v+55)/10))
        # beta = 0.125*np.exp(-(v+65)/80)
        
        dw = np.random.normal(0, np.sqrt(self.dt), size = x.size)

        zeros = np.zeros(x.size)

        langevin = np.where((x>0)&(x<1),np.sqrt(alpha*(1 - x) + beta*x)
            *0.1*np.exp(-0.5/(1 - (2*x - 1)**2)), zeros)

        return (alpha*(1.0 - x) - beta*x)*self.dt + self.sigma*langevin*dw

    def stochastic_chemical_transmitter(self, x, v):

        """ 
        
        This method transforms the Chemical Neurotransmitter gates function into
        stochastic functions

        Args:

            alpha: ion transfer rate
            beta: ion transfer rate
            x: dimenssionless variable that represents the chemical neurontrasnmitters proportion (y) 
        
        """
        alpha = 5./(1 + np.exp(-0.2*(v - 2.)))
        
        beta = 0.18
        
        
        dw = np.random.normal(0, np.sqrt(self.dt), size = x.size)

        zeros = np.zeros(x.size)

        langevin = np.where((x>0)&(x<1),np.sqrt(alpha*(1 - x) + beta*x)
            *0.1*np.exp(-0.5/(1 - (2*x - 1)**2)), zeros)


        return (alpha*(1.0 - x) - beta*x)*self.dt + self.sigma*langevin*dw



    def membrane_potential(self, v, m, n, h, current):

        """ 
        
        This method transforms the membrane potential function into a stochastic differential equation

        Args:

            V:  membrane potential (V)
            M, N, and H: ion particles transfer rate
            I: electrical current
        
        """
        
        dw = np.random.normal(0,np.sqrt(self.dt), size = v.size)
        

        stochastic = HodgkinHuxley.potential(self,v, m, n, h, current)*self.dt + self.sigma_external*dw
        
        return stochastic
