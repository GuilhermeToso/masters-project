""" 

Author: Guilherme M. Toso
Title: spikes.py
Project: Semi-Supervised Learning Using Competition for Neurons' Synchronization

Description: 

    This module contain several methods that can perform some trajectories analyses

"""

""" Dependencies """
import numpy as np

__all__ = ['discrete_potential', 'get_peak_ind', 'get_periods', 'get_spikes_periods']

def discrete_potential(function, threshold):
    """ 
    Returns the trajectories array as an array with 
    1 (V > threshold) and 0 (otherwise)

    Args:

        function: array of the potential trajectories in time of all neurons 
                  (shape: neurons, time)
        theshold: value that indicates there is a spike     
        
     """

    return np.where(function >= threshold, 1, 0)


def get_peak_ind(discrete_array):

    """ 
    Get the indexes of the potential peaks 
    
    Args:

        discrete_array: trajectory of neuron i of 0 and 1 values
        i: the neuron index
    
    """

    indexes = [j for j in range(discrete_array.size) if discrete_array[j-1]==0 and\
         discrete_array[j]==1]

    return indexes

def get_periods(indexes, step):

    """ 
    
    Return the periods of thetrajectory of neuron i
    
    Args:

        indexes: indexes of the peaks of the neuron i
        step: step time
        i: the ith neuron index
    
    
    """

    period = np.array([indexes[j+1] - indexes[j] for j in range(len(indexes) -1)])*step

    return period


def get_spikes_periods(function, threshold, step):

    """ Get the spike indexes and the periods of all neurons """

    spikes = discrete_potential(function, threshold)

    index_list = []
    periods_list = []

    for neuron in range(len(function)):

        indexes = get_peak_ind(spikes[neuron])
        
        periods = get_periods(indexes, step)
        if len(indexes) > 1 and len(periods) > 0:

            index_list.append(indexes)
            periods_list.append(periods)

    return index_list, periods_list



    