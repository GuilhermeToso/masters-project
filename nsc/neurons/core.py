

""" 
Core
====

This module offer inclosed kernels' function for the construction of a Generalised Linear Model. The kernels
are organized into six categories:

1. Fire Probability
2. Internal Distribution
3. Rate Constant
4. Refractory
5. Response
6. Survivor

"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import inspect
import math
import scipy.special as ss
import sys

__all__=['negative_refractory','iaf_refractory','threshold_refractory','mean_lifetime','delay_response','normal_delay_response', 'gamma_response',\
    'moto_response', 'alpha_response','linear','senoidal','exponential','rate_constant','fire_probability','survivor','interval_dist']

def negative_refractory(s,eta_0,delta,tau):
    
    """ 
    .. py::function:

    This is the Negative Refractory Kernel defined by:
    
    .. math::
        Kernel(s) = \Biggl \lbrace 
        {
            1/\\Delta t, \\text{ if } 0 \leq s \leq \\Delta t
            \\atop
            -\\eta_{0}e^{(-s/\\tau)}, s \gt \\Delta t
        }
    

    :param s: The time difference :math:`s` between the actual (`t`) and the last fire time of the :math:`i^{th}` neuron (:math:`t_{i}^{f}`). Which is :math:`s=(t-t_{i}^{f})`
    :type s: numpy.ndarray
    :param eta_0: The hyperpolarization parameter :math:`\\eta_{0}`.
    :type eta_0: int, float or numpy.ndarray
    :param delta: The delay time constant :math:`\\Delta t`
    :type delta: int, float or numpy.ndarray
    :param tau: The membrane time constant :math:`\\tau`.
    :type tau: int, float or numpy.ndarray

    """

    heavside_1 = np.where((time>=0)&(time<=delta),1,0)
    heavside_2 = np.where(time>delta,-eta_0,0)

    value = heavside_2*np.exp(-s/tau) + heavside_1/delta

    return value

def iaf_refractory(s,tau,r,i0):

    """ 
    .. py::function:

    This is the Integrate and Fire Refractory Kernel defined by:

    .. math::
        Kernel(s) = RI_{0}\left[1 - e^{-s/\\tau}\\right]
    
    :param s: The time difference :math:`s` between the actual (`t`) and the last fire time of the :math:`i^{th}` neuron (:math:`t_{i}^{f}`). Which is :math:`s=(t-t_{i}^{f})`
    :type s: numpy.ndarray
    :param tau: The membrane time constant :math:`\\tau`.
    :type tau: int, float or numpy.ndarray
    :param r: The input resistance :math:`R`
    :type r: int, float or numpy.ndarray
    :param i0: The input constant current.
    :type i0: int, float or numpy.ndarray
    
    """

def threshold_refractory(s, V_l, a, V_r,tau,r,h):

    """ 
    .. py::function:
    
    This is the Threshold Refractory Kernel defined by:

    .. math::

        Kernel(s) = \Biggl \lbrace
        {
            A(V_{l}-V_{r})e^{- \\frac { (s-\\tau) } { 2 } + s^{r}}, \\text{ if } 0 \lt s \lt \\tau
            \\atop
            - \\frac { 1 } { h } e^{- \\frac { (s-\\tau) } { 2 } + s^{r}}, \\text{ if } s \gt \\tau
        }

    :param s: The time difference :math:`s` between the actual (`t`) and the last fire time of the :math:`i^{th}` neuron (:math:`t_{i}^{f}`). Which is :math:`s=(t-t_{i}^{f})`
    :type s: numpy.ndarray
    :param V_l: This is the theshold potential :math:`V_{l}`
    :type V_l: int, float or numpy.ndarray
    :param a: This is the action potential amplitude :math:`A`
    :type a: int, float or numpy.ndarray
    :param V_r: This is the rest potential :math:`V_{r}`
    :type V_r: int, float or nupy.ndarray
    :param tau: The membrane time constant :math:`\\tau`.
    :type tau: int, float or numpy.ndarray
    :param r: This is the refractory control parameter
    :type r: int, float or numpy.ndarray
    :param h: This is the hyperpolarization control parameter
    :type h: int, float or numpy.narray
    
    """

    heavside = np.where((s>0)&(s<tau),-a,0)
    heavside = np.where(s>tau,1/h*(V_l-V_r),heavside)
    val = -heavside*(V_l-V_r)*np.exp(-(s-tau)/2 + s**r)

    return val


def mean_lifetime(s,tau):

    """ 
    .. py::function:

    This is the Mean Lifetime Kernel defined by:
    
    .. math::
        Kernel(s) = \\theta(s)e^{(-s/\\tau)}

    Where:

    .. math::
        \\theta(s) = \Biggl \lbrace
        {
        1,\\text{ if }s\geq0
        \\atop
        0,\\text{ otherwise}
        }    
    
    :param s: The time difference :math:`s` between the actual (`t`) and the last fire time of the :math:`i^{th}` neuron (:math:`t_{i}^{f}`). Which is :math:`s=(t-t_{i}^{f})`
    :type s: numpy.ndarray
    :param tau: The membrane time constant :math:`\\tau`.
    :type tau: int, float or numpy.ndarray
    """

    heavside = np.where(s>=0,1,0)

    return heavside*np.exp(-s/tau)

def delay_response(s,delta,tau):

    """ 
    .. py::function:

    This is the Delay Response Kernel defined by:
    
    .. math::
        Kernel(s) = \\theta(s)e^{(-(s-\\Delta t)/\\tau)}

    Where:

    .. math::
        \\theta(s) = \Biggl \lbrace
        {
        1,\\text{ if }s\geq\\Delta t
        \\atop
        0,\\text{ otherwise}
        }    
    
    :param s: The time difference :math:`s` between the actual (`t`) and the last fire time of the :math:`i^{th}` neuron (:math:`t_{i}^{f}`). Which is :math:`s=(t-t_{i}^{f})`
    :type s: numpy.ndarray
    :param delta: The transmissision time delay :math:`\\Delta t`
    :type delta: int, float or numpy.ndarray
    :param tau: The membrane time constant :math:`\\tau`.
    :type tau: int, float or numpy.ndarray

    """

    heavside = np.where(s>delta,1,0)
    #print(heavside)
    return heavside*np.exp(-(s-delta)/tau)

def normal_delay_response(s,delta,tau):

    """ 
    .. py::function:

    This is the Normal Delay Response Kernel defined by:
    
    .. math::
        Kernel(s) = \\theta(s)e^{(-(s-\\Delta t)/\\tau)}

    Where:

    .. math::
        \\theta(s) = \Biggl \lbrace
        {
        1,\\text{ if }s\geq\\Delta t
        \\atop
        N(0,0.1),\\text{ otherwise}
        }    
    
    :param s: The time difference :math:`s` between the actual (`t`) and the last fire time of the :math:`i^{th}` neuron (:math:`t_{i}^{f}`). Which is :math:`s=(t-t_{i}^{f})`
    :type s: numpy.ndarray
    :param delta: The transmissision time delay :math:`\\Delta t`
    :type delta: int, float or numpy.ndarray
    :param tau: The membrane time constant :math:`\\tau`.
    :type tau: int, float or numpy.ndarray
    """
    heavside = np.where(s>delta,1,np.random.normal(0.5,0.1))
    return heavside*np.exp(-(s-delta)/tau)

def gamma_response(s,alpha, beta):

    """ 
    .. py::function:

    This is the Gamma Distribution Response Kernel defined by:
    
    .. math::
        Kernel(s) = \\theta(s) \\frac { \\beta^{\\alpha}s^{\\alpha-1}e^{-\\beta s} } { \\Gamma(\\alpha) }

    Where:

    .. math::
        \\theta(s) = \Biggl \lbrace
        {
        1,\\text{ if }s\geq 0
        \\atop
        0,\\text{ otherwise}
        }    
    
    :param s: The s difference :math:`s` between the actual (`t`) and the last fire time of the :math:`i^{th}` neuron (:math:`t_{i}^{f}`). Which is :math:`s=(t-t_{i}^{f})`
    :type s: numpy.ndarray
    :param alpha: The shape of gamma distribution :math:`\\alpha`
    :type alpha: int, float or numpy.ndarray
    :param beta: The rate of gamma distribution :math:`\\beta`.
    :type beta: int, float or numpy.ndarray
    """

    heavside = np.where(s>0,1,0)

    return heavside*(beta**alpha)*(s**(alpha-1))*(np.exp(-beta*s))/ss.gamma(alpha)

def moto_response(s,u,r,tau_rec,tau_m):

    """ 
    .. py::function:

    This is the Motoneuron Response Kernel defined by:

    .. math::

        Kernel(s,u) = \\frac { R } { \\tau_{m} }\left[1 - e^{(-u/\\tau_{rec})}\\right]e^{(-s/\\tau_{m})}\\theta(s)\\theta(u-s)
    
    Where:

    .. math::

        \\theta(x) = \Biggl \lbrace
        {
            1, \\text{ if } x \gt 0
            \\atop
            0, \\text{ otherwise }
        }
    such that :math:`x` is equal to :math:`s` or :math:`u-s`
    
    :param s: The time difference :math:`s` between the actual (`t`) and the last fire time of the :math:`i^{th}` neuron (:math:`t_{i}^{f}`). Which is :math:`s=(t-t_{i}^{f})`
    :type s: numpy.ndarray
    :param u: The time difference :math:`u` between the actual (`t`) and the last fire time of the actual neuron (:math:`\\hat{t}`). Which is :math:`u=(t-\\hat{t})`
    :type u: numpy.ndarray
    :param r: Input resistance :math:`R`.
    :type r: int, float or numpy.ndarray
    :param tau_rec: Response recovery time :math:`\\tau_{rec}`
    :type tau_rec: int, float or numpy.ndarray
    :param tau_m: Membrane response time :math:`\\tau_{m}`
    :type tau_m: int, float or np.ndarray
    
    """
    heaviside = np.where(u>=0,1,0)
    heaviside_2 = np.where(s>=0,1,0)

    return heaviside*heaviside_2*(r/tau_m)*(1-np.exp(-u/tau_rec))*(np.exp(-s/tau_m))

def alpha_response(s,delta,tau):

    """ 
    .. py::function:

    This is the Normal Delay Response Kernel defined by:
    
    .. math::
        Kernel(s) = \\theta(s-\\Delta t)\left(\\frac { s-\\Delta t} { \\tau^{2}}\\right)e^{(-(s-\\Delta t)/\\tau)}

    Where:

    .. math::
        \\theta(s-\\Delta) = \Biggl \lbrace
        {
        1,\\text{ if }s-\\Delta\gt\\Delta
        \\atop
        0,\\text{ otherwise}
        }    
    
    :param s: The time difference :math:`s` between the actual (`t`) and the last fire time of the :math:`i^{th}` neuron (:math:`t_{i}^{f}`). Which is :math:`s=(t-t_{i}^{f})`
    :type s: numpy.ndarray
    :param delta: The transmissision time delay :math:`\\Delta t`
    :type delta: int, float or numpy.ndarray
    :param tau: The membrane time constant :math:`\\tau`.
    :type tau: int, float or numpy.ndarray
    """  

    heavside = np.where(s>=delta,1,0)
    return heavside*((s-delta)/tau**2)*np.exp(-(s-delta)/tau)


""" External Current Kernel """
def linear(I,w):
    """ 
    This is the Linear Kernel for External Current defined by:

    .. math::
        I(t+1) =I(t) + wI(t)

    :param I: External current.
    :type I: int, float or numpy.ndarray
    :param w: weight of the input current.
    :type w: int, float or numpy.ndarray
    """

    return I+w*I

def senoidal(I,w,t):
    """ 
    This is the Senoidal Kernel for External Current defined by:

    .. math::
        I(t+1) =I(t) + wI(t)sin(t)

    :param I: External current.
    :type I: int, float or numpy.ndarray
    :param w: weight of the input current.
    :type w: int, float or numpy.ndarray
    :param t: time
    :type t: int or float
    
    """

    
    
    return I+w*I*np.sin(t)

def exponential(I,w,t):
    """ 
    This is the Senoidal Kernel for External Current defined by:

    .. math::
        I(t+1) = I(t) + wI(t)e^{t}

    :param I: External current.
    :type I: int, float or numpy.ndarray
    :param w: weight of the input current.
    :type w: int, float or numpy.ndarray
    :param t: time
    :type t: int or float
    
    """


    return I+w*I*np.exp(t)

""" Rate Constant Kernels """

def rate_constant(V,V_l,beta,tau):

    """ 
    This is the Rate Constant Kernel defined by:
    
    .. math::
        \\rho(V) = \\frac { 1 } { \\tau }e^{\\frac { \\beta(V-V_{l}) } { \\tau }}

    :param V: The membrane potential at the isnstant t.
    :type V: int, float or numpy.ndarray
    :param V_l: The rest potential :math:`V_{l}`
    :type V_l: int, float or numpy.ndarray
    :param beta: The noise measurement :math:`\\beta`
    :type beta: int, float or numpy.ndarray
    :param tau: The growth parameter :math:`\\tau`
    :type tau: int, float or numpy.ndarray
    
    """

    p = np.exp(beta*(V-V_l))/tau

    return p

def fire_probability(V,step,V_l,beta,tau):

    """ 
    .. py::function:

    This is the Fire Probability Kernel defined by:

    .. math::

        P_{f}(V) = 1 - e^{-\\delta t\\rho(V)}
    
    :param V: The membrane potential at the isnstant t.
    :type V: int, float or numpy.ndarray
    :param step: The time interval in which a neuron can fire :math:`\\delta t`
    :type step: int, float or numpy.ndarray
    :param V_l: The rest potential :math:`V_{l}`
    :type V_l: int, float or numpy.ndarray
    :param beta: The noise measurement :math:`\\beta`
    :type beta: int, float or numpy.ndarray
    :param tau: The growth parameter :math:`\\tau`
    :type tau: int, float or numpy.ndarray
    
    
    """

    P = 1 - np.exp(-step*rate_constant(V,V_l,beta,tau))

    return P

def survivor(V,delta_t,V_l,beta,tau):

    """ 
    .. py::function:

    This is the Survivor Kernel defined by:

    .. math::

        S(V(s)) = e^{-\\Delta t\\rho(V(s))}
    
    :param V: The membrane potential at the isnstant t.
    :type V: int, float or numpy.ndarray
    :param delta_t: The time interval in which a neuron can not fire :math:`\\Delta t`
    :type delta_t: int, float or numpy.ndarray
    :param V_l: The rest potential :math:`V_{l}`
    :type V_l: int, float or numpy.ndarray
    :param beta: The noise measurement :math:`\\beta`
    :type beta: int, float or numpy.ndarray
    :param tau: The growth parameter :math:`\\tau`
    :type tau: int, float or numpy.ndarray
    
    
    """


    surviving = np.exp(-delta_t*rate_constant(V,V_l,beta,tau))

    return surviving

def interval_dist(V,delta_t,V_l,beta,tau):

    """ 
    .. py::function:

    This is the Interval Distribution Kernel defined by:

    .. math::
        D(V(s)) = S(V(s))\\rho(V(s))
    
    :param V: The membrane potential at the isnstant t.
    :type V: int, float or numpy.ndarray
    :param delta_t: The time interval in which a neuron can not fire :math:`\\Delta t`
    :type delta_t: int, float or numpy.ndarray
    :param V_l: The rest potential :math:`V_{l}`
    :type V_l: int, float or numpy.ndarray
    :param beta: The noise measurement :math:`\\beta`
    :type beta: int, float or numpy.ndarray
    :param tau: The growth parameter :math:`\\tau`
    :type tau: int, float or numpy.ndarray
    
    
    """

    return rate_constant(V,V_L,beta,tau)*survivor(V,delta_t,V_L,beta,tau)
