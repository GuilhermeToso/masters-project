
import numpy as np
import timeit
import sys


def get_peaks_indexes(trajectories, threshold, step):

    """ 
    ..py::function::

    This function get the indexes of the neurons' spikes.

    :param trajectories: The array all the potential values of the n neurons (shape = Iterations, 0, Neurons) 
    :type trajectories: numpy.ndarray
    :param threshold: The threshold value when occurs a spike.
    :type threshold: float.
    :param step: time step.
    :type step: float

    :return: The indexes of the spikes, the times when occurs the spikes, and the periods between each spike    
    :rtype: numpy.ndarray

    """

    if not isinstance(trajectories, np.ndarray):

        raise TypeError("The first argument in the 'get_peaks_indexes' function must be of numpy.ndarray type")

    if trajectories.ndim != 2:

        raise ValueError("The number of dimensions must be 2. (Iterations, Neurons).")


    iters, neurons = trajectories.shape
    
    shifted_ahead_trajecs = trajectories[1:,:]
    shifted_backward_trajecs = trajectories[:-1,:]
    
    indexes_list = []
    times_list = []
    periods_list = []

    for neuron in range(neurons):

        indexes = np.where((shifted_ahead_trajecs[:,neuron]>=threshold)&(shifted_backward_trajecs[:,neuron]<threshold))[0] + 1
        
        times = indexes*step

        indexes_ahead = indexes[1:]
        indexes_backward = indexes[:-1]

        periods = (indexes_ahead - indexes_backward)*step

        indexes_list.append(indexes)
        times_list.append(times)
        periods_list.append(periods)
    
    return indexes_list, times_list, periods_list



def unwrap_static_2(iterations, indexes_list, step, model):

    """ 
    ..py::function::

    This function calculates the phases statically. This is, the phases is found after the calculation of the
    neurons' trajectories.

    :param iterations: Number of iterations in time
    :type iterations: int
    :param indexes_list: A list of n arrays, where n is the amount of neurons and the arrays stores the iteration where occurs a spike
    :type indexes_list: list
    :param step: The step size in time
    :type step: int or float
    
    """
    if not isinstance(iterations,int):

        raise TypeError("The first argument of the function must be of int type, because represents the amount of iterations.")

    if not isinstance(indexes_list,list):

        raise TypeError("The second argument of the function must be of list type, because represents a list that contains arrays of indexes.")
    
    
    if not all(isinstance(array,np.ndarray) for array in indexes_list):

        raise TypeError("All the arrays inside the indexes_list must be of numpy.ndarray type")
    
    if not all(array.dtype == np.int64 for array in indexes_list):

        raise TypeError("The elements in the arrays of the indexes_list must be of int dtype.")
    
    neurons = len(indexes_list)

    phases = np.zeros(shape=(neurons, iterations))

    spikes = np.zeros(shape=(neurons), dtype=int)

    if model == 'HH':
        mean, std = 8.46, 0.65
    elif model =='HR':
        mean, std = 4.29, 0.1475
    elif model=='IAF':
        mean, std = 6.31,0.04
    elif model=='GLM':
        mean, std = 5.44,0.51
    elif model=='Aihara':
        mean, std =2.17,0.17
    elif model=='Rulkov':
        mean,std = 258.51,11.97
    elif model=='Izhikevic':
        mean,std = 23.92,0.56
    elif model=='CNV':
        mean,std = 97.41,0.75  
          
    final_period = np.random.normal(mean, std)/step
    for i in range(neurons):

        indexes = indexes_list[i]
       
        initial = indexes[0]
        final = indexes[-1]
        
        t_0 = indexes[:-1]
        t_f = indexes[1:]

        for j in range(t_f.size):

            time = np.linspace(t_0[j], t_f[j], t_f[j]-t_0[j]+1, dtype=int)
            

            phases[i][time[1:]] = 2*np.pi*j + 2*np.pi*(time[1:] - t_0[j])/(t_f[j] - t_0[j])
            spikes[i] = j

            phases[i][t_0[j]] = 2*np.pi*j
            
        time = np.linspace(t_f[-1], iterations-1, iterations - t_f[-1], dtype=int)

        phases[i][time[1:]] = 2*np.pi*(spikes[i]+1) + 2*np.pi*(time[1:] - t_f[-1])/final_period

        
        spikes[i] = spikes[i] + 1
    return phases
