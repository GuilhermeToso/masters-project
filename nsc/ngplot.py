
""" 

NGPLOT
======

This module offers functions that plot several kinds of graphs

"""

""" Dependencies """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import sys


mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams['text.usetex'] = True

__all__ = ['dim_reduction', 'pca_labeled_unlabeled', 'variance', 'real_classification', 'classified_data', 'phases',
'phases_diff','phases_diff_3D','phase_comparison', 'trajecs_diff_3D', 'neural_activity', 'accuracy_per_seeds', 'accuracy_per_parameters',
'trajectories', 'compare_algs', 'diameters', 'diameters_per_seeds', 'coupling', 'periods_mean']

""" PCA Reduction """

def dim_reduction(var_limit, features):

    """
    .. py::function::

    This function standalize the data (zscore) and then apply the PCA
    method into the data. Transforming it, and finally get the features
    which reachs the variance limit (var_limit) in percentage.
    
    :param var_limit: A value between 0 and 1, that represents the variance limitation.
    :type var_limit: float
    :param features: A numerical dataset of the features.
    :type features: numpy.ndarray or pandas.core.frame.DataFrame 
    :return: A vector of the accumulation sum of the variance (var_cum), the data transformed that achieves the variance limit, the index where the data is splitted, and the vector os the pca variances.
    :rtype: numpy.ndarray, numpy.ndarray, int, numpy.ndarray
    """

    scaler = StandardScaler()

    X = scaler.fit_transform(features)

    pca = PCA()

    pca_data = pca.fit_transform(X)

    pca_var = pca.explained_variance_ratio_

    var_cum = np.cumsum(pca_var)
    
    index = np.where((var_cum >= var_limit))[0][0]
    
    data = pca_data[:,:index]

    return var_cum, data, index, pca_var

def pca_labeled_unlabeled(var_limit, features, unlabel_inds, label_inds, colors, dataset_name):

    """ 
    .. py::function::
    
    3D plot of the first three features after doing PCA 
    (most high variances) 

    :param var_limit: A value between 0 and 1, that represents the variance limitation.
    :type var_limit: float
    :param features: A numerical dataset of the features.
    :type features: numpy.ndarray or pandas.core.frame.DataFrame 
    :param unlabel_inds: A list of the samples's indexes that has no label.
    :type unlabel_inds: list
    :param labels_inds: A dictionaire where the pair key, value represents the label lth (str) and the list of the samples' indexes that are labeled with the lth label.
    :type labels_inds: dict
    :param colors: a list of colors that will represent each label.
    :type colors: list.
    :param dataset_name: The name of the dataset.
    :type dataset_name: str
    :return: A 3d plot of the principal components of the features dataset, with the labeled data colored, and the unlabeled colored by black.

    """
    trasnformed_data = dim_reduction(var_limit, features)
    
    X_reduced = trasnformed_data[1]

    X_3d = X_reduced[:,:3]



    X_3d = pd.DataFrame(data=X_3d,
                        index=range(len(X_3d)),
                        columns=range(3)) 


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_3d.loc[unlabel_inds,0], X_3d.loc[unlabel_inds,1], X_3d.loc[unlabel_inds,2],c='k')
    
    labels_list = list(label_inds.keys())

    for i in range(len(labels_list)):

        ax.scatter3D(X_3d.loc[label_inds[labels_list[i]],0], X_3d.loc[label_inds[labels_list[i]],1], X_3d.loc[label_inds[labels_list[i]],2],c=colors[i],label=labels_list[i])

    ax.set_xlabel('PC 1', fontsize=36, labelpad=40)
    ax.set_ylabel('PC 2', fontsize=36, labelpad=40)
    ax.set_zlabel('PC 3', fontsize=36, labelpad=40)


    variances = np.round(trasnformed_data[3],decimals=4)*100

    plt.annotate(r"Variances (\%)", (1.1, 0.5), xycoords='axes fraction', size=25)
    plt.annotate("PC 1 - {0:6.2f}".format(variances[0]), (1.1, 0.4), xycoords='axes fraction', size=25)
    plt.annotate("PC 2 - {0:6.2f}".format(variances[1]), (1.1, 0.3), xycoords='axes fraction', size=25)
    plt.annotate("PC 3 - {0:6.2f}".format(variances[2]), (1.1, 0.2), xycoords='axes fraction', size=25)
    
    plt.yticks(fontsize=36)
    plt.xticks(fontsize=36)
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(34)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=34)
    plt.title("Dados Rotulados e Não-Rotulados".format(dataset_name),\
        fontsize=24, pad=30)

    plt.show()

def variance(data, target, var = None, cmap = 'viridis', font_box=12):

    """
    .. py::function:
    
    This function returns the variance accumulation plot
    
    Args:

        :param data: A DataFrame object of the data.
        :type data: pandas.core.frame.DataFrame
        :param target: The target column name for prediction
        :type target: str
        :param var: A float value between 0 and 1. Represents the limit of variance accumulation that will appear in the legend.
        :type var: float.
        :param cmap: A colormap name.
        :type cmap: str.

    """
    
    data = data.drop(target, axis=1)

    scaler = StandardScaler()

    X = scaler.fit_transform(data)

    pca = PCA()

    pca_data = pca.fit_transform(X)

    pca_var = np.round(pca.explained_variance_ratio_, decimals=3)
    
    varCum = np.cumsum(pca_var)
    
    str_var = np.round(varCum, 4).astype(str)

    figure, ax = plt.subplots()

    components = [i for i in range(varCum.size)]
    
    ax.plot(components, varCum)
    scatter = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0,vmax=1), cmap=cmap)
    
    ax.scatter(components, varCum, s=100, c=scatter.to_rgba(varCum))

    ax.set_xlabel("Componentes", fontsize=34, labelpad=30)
    ax.set_ylabel(r"Variância (\%)", fontsize=34, labelpad=30)

    ax.tick_params(axis='both', labelsize=30)

    # ax.set_xticks(components)
    # ax.set_xticklabels([str(int(i)) for i in range(varCum.size)])

    ax.set_title("Variância Acumulada por Componentes", fontsize=24)
    
    ax.grid(True)    
    
    normalizer = mpl.colors.Normalize(vmin=0,vmax=1)
    instance = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    colors = []
    if isinstance(var, float) and 0 <= var <= 1:
        size = np.where((varCum >= var))[0][0]
    else:
        size = 3
    for i in range(size):
        color = plt.Rectangle((i,i),0.2,0.2,fc=instance.to_rgba(varCum[i]))
        colors.append(color)

    ax.legend(colors,str_var, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=font_box)
    
    plt.show()
    
def real_classification(var_limit, features, labels, colors, dataset_name):

    """ 
    .. py::function::

    This function plots the real classification of the dataset reduced to 3 principal components.

    :param var_limit: A value between 0 and 1, that represents the variance limitation.
    :type var_limit: float
    :param features: A numerical dataset of the features.
    :type features: numpy.ndarray or pandas.core.frame.DataFrame 
    :param labels: The column in the dataset that contain the labels.
    :type labels: pandas.core.frame.DataFrame
    :param colors: A list of colors that represents each label.
    :type colors: list
    :param dataset_name: The name of the dataset.
    :type dataset_name: str
    :return: A 3d plot of the principal components of the features dataset, with the labeled data colored.
    """



    trasnformed_data = dim_reduction(var_limit, features)
    
    X_reduced = trasnformed_data[1]

    X_3d = X_reduced[:,:3]



    X_3d = pd.DataFrame(data=X_3d,
                        index=range(len(X_3d)),
                        columns=range(3)) 

    uniques = labels.unique()

    
    fig1 = plt.figure()
    ax1 = plt.axes(projection='3d')

    

    for i in range(len(colors)):

        indexes = labels.index[labels == uniques[i]].tolist()

        ax1.scatter3D(X_3d.loc[indexes,0], X_3d.loc[indexes,1], X_3d.loc[indexes,2],c=colors[i],label='{}'.format(uniques[i]))
    

    ax1.set_xlabel('PC 1', fontsize=36, labelpad=40)
    ax1.set_ylabel('PC 2', fontsize=36, labelpad=40)
    ax1.set_zlabel('PC 3', fontsize=36, labelpad=40)


    variances = np.round(trasnformed_data[3],decimals=4)*100
    plt.annotate(r"Variância (\%)", (1.1, 0.5), xycoords='axes fraction', size=25)
    plt.annotate("PC 1 - {0:6.2f}".format(variances[0]), (1.1, 0.4), xycoords='axes fraction', size=25)
    plt.annotate("PC 2 - {0:6.2f}".format(variances[1]), (1.1, 0.3), xycoords='axes fraction', size=25)
    plt.annotate("PC 3 - {0:6.2f}".format(variances[2]), (1.1, 0.2), xycoords='axes fraction', size=25)
    

    plt.yticks(fontsize=36)
    plt.xticks(fontsize=36)
    for t in ax1.zaxis.get_major_ticks(): t.label.set_fontsize(34)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=34)
    plt.title("Classificação Real".format(dataset_name),\
        fontsize=24, pad=30)

    plt.show()

def classified_data(var_limit, features, label_inds, colors, dataset_name):

    """ 
    .. py::function::        
    
    3D plot of the classified data after using PCA.  
    
    :param var_limit: A value between 0 and 1, that represents the variance limitation.
    :type var_limit: float
    :param features: A numerical dataset of the features.
    :type features: numpy.ndarray or pandas.core.frame.DataFrame 
    :param labels_inds: A dictionaire where the pair key, value represents the label lth (str) and the list of the samples' indexes that are labeled with the lth label.
    :type labels_inds: dict
    :param colors: a list of colors that will represent each label.
    :type colors: list.
    :param dataset_name: The name of the dataset.
    :type dataset_name: str
    :return: A 3d plot of the principal components of the features dataset, with the labeled data colored

    """




    trasnformed_data = dim_reduction(var_limit, features)
    
    X_reduced = trasnformed_data[1]

    X_3d = X_reduced[:,:3]

    X_3d = pd.DataFrame(data=X_3d,
                        index=range(len(X_3d)),
                        columns=range(3)) 


    fig = plt.figure()
    ax = plt.axes(projection='3d')

    labels_list = list(label_inds.keys())

    for i in range(len(labels_list)):

        ax.scatter3D(X_3d.loc[label_inds[labels_list[i]],0], X_3d.loc[label_inds[labels_list[i]],1], X_3d.loc[label_inds[labels_list[i]],2],c=colors[i],label=labels_list[i])

    ax.set_xlabel('PC 1', fontsize=36, labelpad=40)
    ax.set_ylabel('PC 2', fontsize=36, labelpad=40)
    ax.set_zlabel('PC 3', fontsize=36, labelpad=40)


    variances = np.round(trasnformed_data[3],decimals=4)*100
    
    plt.annotate(r"Variância (\%)", (1.1, 0.5), xycoords='axes fraction', size=25)
    plt.annotate("PC 1 - {0:6.2f}".format(variances[0]), (1.1, 0.4), xycoords='axes fraction', size=25)
    plt.annotate("PC 2 - {0:6.2f}".format(variances[1]), (1.1, 0.3), xycoords='axes fraction', size=25)
    plt.annotate("PC 3 - {0:6.2f}".format(variances[2]), (1.1, 0.2), xycoords='axes fraction', size=25)
    
    plt.yticks(fontsize=36)
    plt.xticks(fontsize=36)
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(34)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=34)
    plt.title("Dados Classificados".format(dataset_name),\
        fontsize=24, pad=30)

    plt.show()

def phases(phases_array, colors, step):

    """ 
    ..py::function::

    This function plots the growing phases through time
    
    :param phases_array: The array of the phases of every neuron through time. Shape: neurons, iterations.
    :type phases_array: numpy.ndarray
    :param colors: a list of the colors for every neuron's trajectory.
    :type colors: list
    :param step: time step
    :type step: int or float
    
    
    """

    # if not isinstance(colors, list):

    #     raise TypeError(" The colors argument must be a list ")

    if not isinstance(phases_array, np.ndarray):

        raise TypeError("The phases array must be of type numpy.ndarray")

    if phases_array.ndim != 2:

        raise ValueError("The dimensions of the phases' array must be 2. (Neurons, Iterations).")

    neurons = phases_array.shape[0]
    
    time = np.linspace(0,phases_array.shape[1], num=phases_array.shape[1], dtype=int)*step

    for i in range(neurons):

        plt.plot(time, phases_array[i][:],colors[i], label = r'$\phi_{0}$'.format(i))
    
    plt.xlabel(r't [ms]',fontsize=36, labelpad=10)
    plt.ylabel(r'$\phi_{i}$',fontsize=36, labelpad=10)
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.grid(True)
    plt.title("Fases", fontsize=36)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=34)
    plt.show()

def phases_diff(phase_ref, phases, step,colors = None, cmap = None, set_limit = False):

    """ 
    ..py::function::

    This function calculates difference of the phases trajectories in relation to
    a reference phase trajectory.
    
    :param phase_ref: Phase trajectory reference
    :type phase_ref: numpy.ndarray
    :param phases: Phases trajectories of all neurons.
    :type phases: numpy.ndarray
    :param step: The time step
    :type step: int or float
    :param colors: a list of colors defned by the user
    :type colors: list
    :param cmap: a colormap name
    :type cmap: str
    :param set_limit: Set a limit to the y axis (multiple of pi). Default is False, which means thee is no limit.
    :type set_limit: bool

    """

    if not isinstance(phase_ref, int):

        raise TypeError("The phase reference must be of the type integer.")

    if not isinstance(phases, np.ndarray):

        raise TypeError("The phases must be of the type numpy.ndarray.")

    # if phase_ref.ndim != 1:

    #     raise ValueError("The phase reference dimension must be 1. (Iterations)")

    if phases.ndim != 2:

        raise ValueError("The phases dimension must be 2. (Neurons, Iterations)")

    if (colors != None) and (not isinstance(colors,list)):

        raise TypeError("The colors parameter must be of the type list")

    if (cmap != None) and (not isinstance(cmap,str)):

        raise TypeError("The cmap parameter must be of the type str")

    if not isinstance(set_limit,bool):

        raise TypeError("The set_limit parameter must be of the type bool.")

    if (phases.dtype == np.int64 or phases.dtype == np.float64):

        time = np.arange(phases[phase_ref].size)*step
        neurons = phases.shape[0] - 1
        
        if (colors == None) and (cmap == None):

            c_list = cm.get_cmap('viridis', neurons)

        elif (colors == None) and (cmap != None):

            c_list = cm.get_cmap(cmap, neurons).colors

        elif ((colors != None) and (cmap == None)) or ((colors != None) and (cmap != None)):
        
            c_list = colors

        #y_ticks = np.linspace(0,12.5415926536,9)
        #print("Phase Ref: \n", phase_ref)
        # print(phases.shape)
        # row = np.where(phase_ref == phases)[0][0]
        # print("\n Row: \n", row)
        # sys.exit(0)
        # #print("\n Phases: \n", phases)
        #temp_phases = np.delete(phases, row, 0)
        #print("\n Temporary Phases: \n", temp_phases)
        #print("\n Temporary Phases Shape: ", temp_phases.shape)
        diff_phase = np.abs(phases[phase_ref] - phases)
        diff_phase = np.delete(diff_phase, phase_ref,0)


        #temp_diff = np.zeros((diff_phase.shape[0], diff_phase.shape[1]))            
        for i in range(neurons):

            #diff_phase = np.abs(phase_ref - temp_phases[i])

            #temp_diff[i,:] = diff_phase

            plt.plot(time, diff_phase[i,:], c = c_list[i])
        

        plt.plot(time, np.ones(time.size)*2*np.pi,'r',label = r'$2\pi$')
        
        
        if set_limit == False:

            plt.yticks(fontsize=36)

            #plt.ylim((0,12.5415926536))
            #plt.yticks(y_ticks,('0',r'$\frac{\pi}{2}$',r'$\pi$',r'$\frac{3\pi}{2}$',r'$2\pi$',r'$\frac{5\pi}{2}$',r'$3\pi$',r'$\frac{7\pi}{2}$',r'$4\pi$'),fontsize=34)
        elif set_limit == True:

            ymax = np.max(diff_phase)
            
            new_limit = int(ymax/np.pi)*np.pi + np.pi
            
            
            if 0 < new_limit <= 2*np.pi:

                y_ticks = np.linspace(0,2*np.pi,5)
                plt.ylim((0,2*np.pi))
                plt.yticks(y_ticks,('0',r'$\frac{\pi}{2}$',r'$\pi$',r'$\frac{3\pi}{2}$',r'$2\pi$'), fontsize=36)
            
            elif 2*np.pi < new_limit <= 4*np.pi:
            
                y_ticks = np.linspace(0,4*np.pi,9)
                plt.ylim((0,4*np.pi))
                plt.yticks(y_ticks,('0',r'$\frac{\pi}{2}$',r'$\pi$',r'$\frac{3\pi}{2}$',r'$2\pi$',r'$\frac{5\pi}{2}$',r'$3\pi$',r'$\frac{7\pi}{2}$',r'$4\pi$'),fontsize=36)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=34)
        
            elif 4*np.pi < new_limit <= 10*np.pi:

                y_ticks = np.linspace(0,10*np.pi,11)
                plt.ylim((0,10*np.pi))
                plt.yticks(y_ticks,('0',r'$\pi$',r'$2\pi$',r'$3\pi$',r'$4\pi$',r'$5\pi$',r'$6\pi$',r'$7\pi$',r'$8\pi$',r'$9\pi$',r'$10\pi$'),fontsize=36)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=34)
        
            elif 10*np.pi < new_limit <= 20*np.pi:

                y_ticks = np.linspace(0,20*np.pi,11)
                plt.ylim((0,20*np.pi))
                plt.yticks(y_ticks,('0',r'$2\pi$',r'$4\pi$',r'$6\pi$',r'$8\pi$',r'$10\pi$',r'$12\pi$',r'$14\pi$',r'$16\pi$',r'$18\pi$',r'$20\pi$'),fontsize=36)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=34)
        

        
        plt.xlabel(r't [ms]',fontsize=36)
        plt.ylabel(r'$|\phi_{i} - \phi_{j}|$',fontsize=36, labelpad=34, rotation=90)
        plt.xticks(fontsize=36)
        plt.title("Diferença de fases", fontsize=36)
        plt.grid(True)
        plt.show()

    else:

        raise TypeError("The phases and phase reference must be of numerical dtype (np.int64 or np.float64).")

def phases_diff_3D(phase_ref, phases, step,cmap = None, set_limit = False, facecolor='r'):

    """ 
    ..py::function::

    This function calculates difference of the phases trajectories in relation to
    a reference phase trajectory.
    
    :param phase_ref: Phase trajectory reference
    :type phase_ref: numpy.ndarray
    :param phases: Phases trajectories of all neurons.
    :type phases: numpy.ndarray
    :param step: The time step
    :type step: int or float
    :param colors: a list of colors defned by the user
    :type colors: list
    :param cmap: a colormap name
    :type cmap: str
    :param set_limit: Set a limit to the y axis (multiple of pi). Default is False, which means thee is no limit.
    :type set_limit: bool

    """

    if not isinstance(phase_ref, int):

        raise TypeError("The phase reference must be of the type integer.")

    if not isinstance(phases, np.ndarray):

        raise TypeError("The phases must be of the type numpy.ndarray.")

    # if phase_ref.ndim != 1:

    #     raise ValueError("The phase reference dimension must be 1. (Iterations)")

    if phases.ndim != 2:

        raise ValueError("The phases dimension must be 2. (Neurons, Iterations)")

    if (cmap != None) and (not isinstance(cmap,str)):

        raise TypeError("The cmap parameter must be of the type str")

    if not isinstance(set_limit,bool):

        raise TypeError("The set_limit parameter must be of the type bool.")

    if (phases.dtype == np.int64 or phases.dtype == np.float64):

        neurons = phases.shape[0] - 1
        print("Neurons at Phases: ", neurons)

        diff = np.arange(neurons)

        time = np.arange(phases[phase_ref].size)*step

        diff1, time1 = np.meshgrid(diff, time)

        if cmap == None:

            cmap = cm.get_cmap('viridis')

        elif cmap != None:

            cmap = cm.get_cmap(cmap)


        #y_ticks = np.linspace(0,12.5415926536,9)

        #row = np.where(phase_ref == phases)[0][0]

        #temp_phases = np.delete(phases, row, 0)

        temp_diff = np.abs(phases[phase_ref] - phases)

        limit = np.zeros((neurons, phases[phase_ref].size)) + 2*np.pi

        temp_diff = np.delete(temp_diff,phase_ref,0)

        """ Create figure """
        fig = plt.figure()

        """ Create axis """
        ax = fig.gca(projection='3d')
        surface = ax.plot_surface(diff1.T, time1.T, temp_diff, cmap=cmap, edgecolor=None)
        
        surface_2pi = ax.plot_surface(diff1.T, time1.T, limit, color='#e41a1c',linewidth=0, antialiased=False,alpha=0.2, facecolor='#e41a1c', shade=False)

        cbar = fig.colorbar(surface, shrink=0.5, aspect=5, pad=0.1)
        cbar.ax.tick_params(labelsize=24) 
        ax.set_xlabel('Osciladores', fontsize=34, labelpad=40)
        ax.set_ylabel('t', fontsize=30, labelpad=40)
        #ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(r'$|\phi_{i} - \phi_{j}|$',fontsize=34, labelpad=40)

        new_time = np.linspace(0,time[-1],(5), dtype=int) + 1
        time_ticks = list(new_time.copy().astype(str))
        time_list = list(new_time)

        # oscilators = np.linspace(0,neurons-1,(neurons), dtype=int)
        # oscilators_ticks = list(oscilators.copy().astype(str))
        # oscilators_list = list(oscilators)

        plt.yticks(time_list, time_ticks, fontsize=34)
        if neurons <= 7:
            locs = [i for i in range(neurons)]
            labels = [str(i) for i in range(neurons)]
        elif neurons > 7:

            stepsize = int(neurons/7)+1
            actual = 0
            locs = []
            labels = []
            for i in range(8):
                locs = locs + [actual]
                labels = labels + [str(actual)]
                actual = actual + stepsize
                if actual > neurons:
                    actual = neurons+1
                    locs = locs + [actual]
                    labels = labels + [str(actual)]
                    break
        plt.xticks(locs, labels, fontsize=34)
        for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(34)
        ax.view_init(20,-60)
        color  = [plt.Rectangle((0, 0), 1, 1, fc="r")]
        labels = [r'$|\phi_{i}- \phi_{j}| = 2\pi$']
        plt.legend(color, labels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=34)

        plt.show()

def phase_comparison(phase_ref, phases, step,cmap = None, set_limit = False, facecolor='r'):

    """ 
    ..py::function::

    This function calculates difference of the phases trajectories in relation to
    a reference phase trajectory.
    
    :param phase_ref: Phase trajectory reference
    :type phase_ref: numpy.ndarray
    :param phases: Phases trajectories of all neurons.
    :type phases: numpy.ndarray
    :param step: The time step
    :type step: int or float
    :param colors: a list of colors defned by the user
    :type colors: list
    :param cmap: a colormap name
    :type cmap: str
    :param set_limit: Set a limit to the y axis (multiple of pi). Default is False, which means thee is no limit.
    :type set_limit: bool

    """

    if not isinstance(phase_ref, int):

        raise TypeError("The phase reference must be of the type integer.")

    if not isinstance(phases, np.ndarray):

        raise TypeError("The phases must be of the type numpy.ndarray.")

    # if phase_ref.ndim != 1:

    #     raise ValueError("The phase reference dimension must be 1. (Iterations)")

    if phases.ndim != 3:

        raise ValueError("The phases dimension must be 3. (Neurons, Trials, Iterations)")

    if (cmap != None) and (not isinstance(cmap,str)):

        raise TypeError("The cmap parameter must be of the type str")

    if not isinstance(set_limit,bool):

        raise TypeError("The set_limit parameter must be of the type bool.")

    if (phases.dtype == np.int64 or phases.dtype == np.float64):

        neurons = phases.shape[0]-1
        epochs = phases.shape[1]
        #print("Neurons at Phases: ", neurons)

        trials = np.arange(epochs)

        time = np.arange(phases.shape[2])*step

        trials_grid, time_grid = np.meshgrid(trials, time)

        if cmap == None:

            cmap = cm.get_cmap('viridis')

        elif cmap != None:

            cmap = cm.get_cmap(cmap)

        cmap = ['viridis', 'plasma', 'inferno']
        #y_ticks = np.linspace(0,12.5415926536,9)

        #row = np.where(phase_ref == phases)[0][0]

        #temp_phases = np.delete(phases, row, 0)

        temp_diff = np.abs(phases[phase_ref] - phases)

        limit = np.zeros((epochs, phases.shape[2])) + 2*np.pi

        temp_diff = np.delete(temp_diff,phase_ref,0)

        """ Create figure """
        fig = plt.figure()

        """ Create axis """
        ax = fig.gca(projection='3d')

        for i in range(neurons):
            ax.plot_surface(trials_grid.T, time_grid.T, temp_diff[i,:,:], cmap=cm.get_cmap(cmap[i]), edgecolor=None)
        
        surface_2pi = ax.plot_surface(trials_grid.T, time_grid.T, limit, color='#e41a1c',linewidth=0, antialiased=False,alpha=0.2, facecolor='#e41a1c', shade=False)

        # cbar = fig.colorbar(surface, shrink=0.5, aspect=5, pad=0.1)
        # cbar.ax.tick_params(labelsize=24) 
        ax.set_xlabel('Osciladores', fontsize=34, labelpad=40)
        ax.set_ylabel('t', fontsize=30, labelpad=40)
        #ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(r'$|\phi_{i} - \phi_{j}|$',fontsize=34, labelpad=40)

        new_time = np.linspace(0,time[-1],(5), dtype=int) + 1
        time_ticks = list(new_time.copy().astype(str))
        time_list = list(new_time)

        # oscilators = np.linspace(0,neurons-1,(neurons), dtype=int)
        # oscilators_ticks = list(oscilators.copy().astype(str))
        # oscilators_list = list(oscilators)

        # plt.yticks(time_list, time_ticks, fontsize=34)
        # if neurons <= 7:
        #     locs = [i for i in range(neurons)]
        #     labels = [str(i) for i in range(neurons)]
        # elif neurons > 7:

        #     stepsize = int(neurons/7)+1
        #     actual = 0
        #     locs = []
        #     labels = []
        #     for i in range(8):
        #         locs = locs + [actual]
        #         labels = labels + [str(actual)]
        #         actual = actual + stepsize
        #         if actual > neurons:
        #             actual = neurons+1
        #             locs = locs + [actual]
        #             labels = labels + [str(actual)]
        #             break
        plt.xticks(fontsize=34)
        for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(34)
        ax.view_init(20,-60)
        color  = [plt.Rectangle((0, 0), 1, 1, fc="r")]
        labels = [r'$|\phi_{i}- \phi_{j}| = 2\pi$']
        plt.legend(color, labels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=34)

        plt.show()

def trajecs_diff_3D(trajec_ref, trajecs, step, cmap = None, set_limit = None):

    """ 
    ..py::function::

    This function calculates difference of the trajectories in relation to
    a reference trajectory.
    
    :param trajec_ref: Trajectory reference
    :type trajec_ref: numpy.ndarray
    :param trajecs: Trajectories of all neurons.
    :type trajecs: numpy.ndarray
    :param step: The time step
    :type step: int or float
    :param colors: A list of colors defned by the user
    :type colors: list
    :param cmap: A colormap name
    :type cmap: str
    :param set_limit: Set a limit to the y axis. Default is None.
    :type set_limit: int or float

    """

    if not isinstance(trajec_ref, int):

        raise TypeError("The trajectory reference must be of the type integer.")

    if not isinstance(trajecs, np.ndarray):

        raise TypeError("The trajecs must be of the type numpy.ndarray.")

    # if trajec_ref.ndim != 1:

    #     raise ValueError("The trajectory reference dimension must be 1. (Iterations)")

    if trajecs.ndim != 2:

        raise ValueError("The trajecs dimension must be 2. (Neurons, Iterations)")

    if (cmap != None) and (not isinstance(cmap,str)):

        raise TypeError("The cmap parameter must be of the type str")

    if ((set_limit != None) and (not isinstance(set_limit,float))):

        raise TypeError("The set_limit parameter must be of the type float.")

    if (trajecs.dtype == np.int64 or trajecs.dtype == np.float64):

        neurons = trajecs.shape[0] - 1
        print("Neurons at Trajecs ", neurons)
        
        diff = np.arange(neurons)

        time = np.arange(trajecs[trajec_ref].size)*step

        diff1, time1 = np.meshgrid(diff, time)

        if cmap == None:

            cmap = cm.get_cmap('viridis')

        elif cmap != None:

            cmap = cm.get_cmap(cmap)


        # row = np.where(trajecs[trajec_ref] == trajecs)[0][0]

        # temp_trajecs = np.delete(trajecs, row, 0)

        temp_diff = np.abs(trajecs[trajec_ref] - trajecs)
        temp_diff = np.delete(temp_diff, trajec_ref, 0)


        """ Create figure """
        fig = plt.figure()

        """ Create axis """
        ax = fig.gca(projection='3d')
        
        surface = ax.plot_surface(diff1.T, time1.T, temp_diff, cmap=cmap, edgecolor=None)
      
        cbar = fig.colorbar(surface, shrink=0.5, aspect=5, pad=0.1)
        cbar.ax.tick_params(labelsize=34) 
        ax.set_xlabel('Osciladores', fontsize=34, labelpad=40)
        ax.set_ylabel('t', fontsize=30, labelpad=40)
        ax.set_zlabel(r'$|V_{i} - V_{j}|$',fontsize=34, labelpad=40)

        new_time = np.linspace(0,time[-1],(5), dtype=int) + 1
        time_ticks = list(new_time.copy().astype(str))
        time_list = list(new_time)

        plt.yticks(time_list, time_ticks, fontsize=34)
        if neurons <= 7:
            locs = [i for i in range(neurons)]
            labels = [str(i) for i in range(neurons)]
        elif neurons > 7:

            stepsize = int(neurons/7)+1
            actual = 0
            locs = []
            labels = []
            for i in range(8):
                locs = locs + [actual]
                labels = labels + [str(actual)]
                actual = actual + stepsize
                if actual > neurons:
                    actual = neurons+1
                    locs = locs + [actual]
                    labels = labels + [str(actual)]
                    break
        plt.xticks(locs, labels, fontsize=34)
        for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(34)
        ax.view_init(20,-60)
        
        plt.show()

def neural_activity(time_array, neurons_array, time_max, cmap = 'viridis', colors = None, **kwargs):
    
    """ 
    .. py::function::
    
    This function plots the neural activity of neurons per time (which neuron n fires at time t).

    :param time_array: The time array (x-axis) of fires' times of every neuron (neuron x time)
    :type time_array: list of numpy.ndarrays.
    :param neurons_array: the neurons array (y-axis) of which neuron index is the neural activity.
    :type neurons_array: list of numpy.ndarrays.
    :param time_max: time when ends the iteration
    :type time_max: int or float
    

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *cmap* (``str``) --
        A string value that represents the color to be used.

        * *colors* (``list``) --
        A list of colors that represents each group synchronized.

    """

    if not isinstance(time_array,list):

        raise TypeError("The time array must be a list of numpy.ndarrays.")
    
    if not isinstance(neurons_array,list):

        raise TypeError("The neurons array must be a list of numpy.ndarrays.")

    if len(time_array) != len(neurons_array):

        raise ValueError("Both the time and neurons arrays must have the same length.")

    
    neurons = len(neurons_array)

    labeled_inds = kwargs.get("labeled")


    if isinstance(colors,list) or isinstance(colors, np.ndarray):

        if labeled_inds == None:

            raise ValueError("The labeled indexes must not be None, it must be a dictionary.")

        else:

            color = colors

            keys = list(labeled_inds.keys())

            cols = []

            for i in range(len(keys)):

                indexes = labeled_inds[keys[i]]

                for j in range(len(time_array)):

                    if j in indexes:

                        plt.scatter(time_array[j],neurons_array[j], s=100, c=color[i])

                cols.append(plt.Rectangle((i, i), 1, 1, fc=color[i]))

            plt.xlabel("t [ms]", fontsize=36, labelpad=30)
            plt.ylabel("Neurônios", fontsize=36, labelpad=30)

            plt.xlim(0, time_max)

            new_time = np.linspace(0,time_max,(6), dtype=int)
            time_ticks = list(new_time.copy().astype(str))
            time_list = list(new_time)

        
            new_neurons = np.linspace(0,neurons,(6), dtype=int)
            neurons_ticks = list(new_neurons.copy().astype(str))
            neurons_list = list(new_neurons)
            plt.yticks(neurons_list, neurons_ticks, fontsize=36)

            plt.xticks(time_list, time_ticks, fontsize=36)
            
            plt.legend(cols,keys,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=34)

            plt.grid(True)

            plt.show()

    else:

        color = cm.get_cmap(cmap, neurons).colors

        for i in range(neurons):

            plt.scatter(time_array[i],neurons_array[i], s=100, c=color[i])
        
        plt.xlabel("t [ms]", fontsize=36, labelpad=30)
        plt.ylabel("Neurônios", fontsize=36, labelpad=30)

        plt.xlim(0, time_max)

        new_time = np.linspace(0,time_max,(6), dtype=int)
        time_ticks = list(new_time.copy().astype(str))
        time_list = list(new_time)

        new_neurons = np.linspace(0,neurons,(6), dtype=int)
        neurons_ticks = list(new_neurons.copy().astype(str))
        neurons_list = list(new_neurons)
        plt.yticks(neurons_list, neurons_ticks, fontsize=36)
    

        plt.xticks(time_list, time_ticks, fontsize=36)
        
        plt.grid(True)

        plt.show()

def accuracy_per_seeds(seeds_array, accuracy_array, labels, y_limit = True, show_value = True, stds = None, cmap = 'viridis', **kwargs):
    
    """ 
    .. py::function::

    This function calculates the accuracy means and the standard deviations per seeds.

    :param seeds_array: The seeds array
    :type seeds_array: numpy.ndarray
    :param accuracy_array: The accuracy array
    :type accuracy_array: numpy.ndarray
    :param labels: a list with the labels.
    :type labels: list
    
    :param y_limit: If there is a limit (100 %) or not. Default is True.
    :type y_limit: bool
    :param show_value: Show the accuracy per seed. Default is True.
    :type show_value: bool
    :param stds: How many standard deviations will be shown. Default None.
    :type stds: int.
    :param cmap: a string value that represents a colormap
    :type cmap: str
    
    :param \**kwargs:
        See below

    :Keyword Arguments:
        
        * *colors* (``list``) --
        A list of colors that represents each group synchronized.

        * *std_array* (``numpy.ndarray``)
        The array of standard deviations

    """

    if ((not isinstance(seeds_array,np.ndarray))and(not isinstance(accuracy_array,np.ndarray))):

        raise TypeError("Seeds and Accuracy must be of the type numpy.ndarray.")

    if ((not isinstance(y_limit,bool))and(not isinstance(show_value,bool))):

        raise TypeError("Y limit and show_value must be of the type bool.")

    if ((stds != None) and (not isinstance(stds, int))):

        raise TypeError("Stds must be of the type int.")

    colors = kwargs.get("colors")

    std_array = kwargs.get("std_array")

    if colors == None:

        cols = cm.get_cmap(cmap,len(labels)).colors
    
    else:

        cols = colors

    if stds != None:

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_xlabel(r'Nº Sementes', fontsize=36)
        ax.set_ylabel(r'Acurácia', fontsize=36)

        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)

        line = {"linestyle":"-", "linewidth":2, "markeredgewidth":2,\
        "elinewidth":2, "capsize":3}

        for i in range(accuracy_array.shape[0]):

            ax.errorbar(seeds_array, accuracy_array[i], yerr = std_array[i]*stds, \
                **line, color = cols[i], label = labels[i])

        if show_value == True:

            accuracy_array = np.around(accuracy_array, decimals=2)
            for j in range(len(accuracy_array)):
                for i, txt in enumerate(accuracy_array[j]):   
                    ax.annotate(txt, xy=(seeds_array[i], accuracy_array[j][i]),
                    xytext=(seeds_array[i]+0.03,accuracy_array[j][i]+0.3),
                    color=cols[j]) 


        if y_limit == True:

            ax.set_ylim([0,100])

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=34)
        plt.grid(True)
        plt.show()

    elif stds == None:
        
        fig = plt.figure()
        ax = fig.add_subplot(111)


        ax.set_xlabel(r'Nº Sementes', fontsize=36)
        ax.set_ylabel(r'Acurácia', fontsize=36)

        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)

        #line = {"linestyle":"-", "linewidth":2, "markeredgewidth":2,\
        #"capsize":3}

        for i in range(accuracy_array.shape[0]):

            ax.plot(seeds_array, accuracy_array[i], color = cols[i], label = labels[i])

        if show_value == True:

            accuracy_array = np.around(accuracy_array, decimals=2)
            for j in range(len(accuracy_array)):
                for i, txt in enumerate(accuracy_array[j]):   
                    ax.annotate(txt, xy=(seeds_array[i], accuracy_array[j][i]),
                    xytext=(seeds_array[i]+0.03,accuracy_array[j][i]+0.3),
                    color=cols[j]) 


        if y_limit == True:

            ax.set_ylim([0,100])

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=34)
        plt.grid(True)
        plt.show()

def accuracy_per_parameters(alpha_num=None, beta_num=None, alpha_cat=None, beta_cat=None, accuracy = None, most_high = 5, labels = None, colors = None, cmap = 'rainbow'):

    """ 
    
    .. py::function::

    This function plots the accuracy per parameters (alpha numeric, beta numeric, alpha categoric and/or beta categoric).
    
    :param alpha_num: An array of alpha numeric values. By default None.
    :type alpha_num: Accept list and numpy.ndarray
    :param beta_num: An array of beta numeric values. By default None.
    :type beta_num: Accept list and numpy.ndarray
    :param alpha_cat: An array of alpha categoric values. By default None.
    :type alpha_cat: Accept list and numpy.ndarray
    :param beta_cat: An array of beta categoric values. By default None.
    :type beta_cat: Accept list and numpy.ndarray
    :param accuracy: An array of accuracy values. By default None.
    :type accuracy: Accept list and numpy.ndarray
    :param labels: A list with the labels names.
    :type labels: list with strings.
    :param colors: A list with the colors.
    :type colors: list
    :param cmap: The color map used to represent one of the parameters
    :type cmap: str
    
    
    """

    if ((not isinstance(accuracy, list)) and (not isinstance(accuracy, np.ndarray))):

        raise TypeError("Accuracy must be a list or a numpy array of floats with the same shape as \n the parameters set in the function.")

    if (not isinstance(labels, list)):

        raise TypeError("Labels must be a list with the labels names.")


    dictionary = {
        'alpha_num':alpha_num,
        'beta_num':beta_num,
        'alpha_cat':alpha_cat,
        'beta_cat':beta_cat
    }

    dic_names = {
        'alpha_num': r'$\alpha_{n}$',
        'beta_num': r'$\beta_{n}$',
        'alpha_cat':r'$\alpha_{c}$',
        'beta_cat':r'$\beta_{c}$'
    }

    for key, value in dictionary.items():

        if (isinstance(dictionary[key],list)):

            if ((any(type(val) != int for val in dictionary)) or (any(type(val) != float for val in dictionary))):

                raise TypeError("Data type in list must be of type integer or float.")

        elif (isinstance(dictionary[key],np.ndarray)):

            if ((dictionary[key].dtype != 'float64') and (dictionary[key].dtype != 'int32')):

                raise TypeError("Data type in the array must be of type integer or float.")

    
    keys = []
    values = []

    for key,value in dictionary.items():

        if ((isinstance(dictionary[key],list)) or (isinstance(dictionary[key],np.ndarray))):

            keys.append(key)
            values.append(value)

    if len(keys) == 1:

        accuracy = accuracy*100
        
        if (isinstance(colors, list)):

            cols = colors

        else:

            cols = cm.get_cmap(cmap,len(labels)).colors()

        plt.xlabel(dic_names[keys[0]], fontsize=36, labelpad=40)
        plt.ylabel(r'Acurácia (\%)', fontsize = 36, labelpad=40)

        new_accuracy = np.linspace(0,100,(6), dtype=int)
        accuracy_ticks = list(new_accuracy.copy().astype(str))
        accuracy_list = list(new_accuracy)

        new_par = np.linspace(0,values[0][-1]+1,(6), dtype=int)
        par_ticks = list(new_par.copy().astype(str))
        par_list = list(new_par)

        plt.xticks(par_list, par_ticks, fontsize=34)
        plt.yticks(accuracy_list, accuracy_ticks, fontsize = 34)

        for i in range(len(accuracy)):

            plt.scatter(values[0], accuracy[i], s = 50, c = cols[i], marker= 'o')
            plt.plot(values[0], accuracy[i], c = cols[i], label = labels[i])

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=34)
        plt.grid(True)

        plt.show()


    elif len(keys) == 2:
        
        accuracy = accuracy*100

        for i in range(len(labels)):

            fig = plt.figure()

            ax = fig.gca(projection='3d')

            par_1, par_2 = np.meshgrid(values[0], values[1])

            cmap = cm.get_cmap(cmap)

            all_true = np.all(accuracy[i] == accuracy[i][0][0])

            
            if all_true: 

                surface = ax.plot_surface(par_1.T, par_2.T, accuracy[i], cmap=cmap, edgecolor=None, vmin=0, vmax=100)           
                ax.set_zlim(0,100)

                normalize = mpl.colors.Normalize(vmin=0, vmax=100)
    
                sm = mpl.cm.ScalarMappable(norm=normalize, cmap = cmap)

                cbar = fig.colorbar(sm, ticks=np.linspace(0,100,11))
                cbar.ax.set_yticklabels(np.linspace(0,100,11).astype(int).astype(str), fontsize=30)
                cbar.ax.set_ylabel('Acurácia', fontsize=30)

            else:
                surface = ax.plot_surface(par_1.T, par_2.T, accuracy[i], cmap=cmap, edgecolor=None, vmin=int(accuracy[i].min()), vmax=100)           
                
                normalize = mpl.colors.Normalize(vmin=int(accuracy[i].min()), vmax=100)
    
                sm = mpl.cm.ScalarMappable(norm=normalize, cmap = cmap)

                cbar = fig.colorbar(sm, ticks=np.linspace(int(accuracy[i].min()),100,11))
                cbar.ax.set_yticklabels(np.round(np.linspace(int(accuracy[i].min()),100,11),decimals=2).astype(str), fontsize=30)
                cbar.ax.set_ylabel('Acurácia', fontsize=30)

                ax.set_zlim(int(accuracy[i].min()), 100)

            
            ax.set_xlabel(dic_names[keys[0]], fontsize=34, labelpad=40)
            ax.set_ylabel(dic_names[keys[1]], fontsize=34, labelpad=40)
            ax.set_zlabel(r'Acurácia (\%)', fontsize=34, labelpad=40)

            new_par_1 = np.linspace(0, values[0][-1], (6), dtype=int)
            new_par_1_ticks = list(new_par_1.copy().astype(str))
            new_par_1_list = list(new_par_1)

            new_par_2 = np.linspace(0, values[1][-1], (6), dtype=int)
            new_par_2_ticks = list(new_par_2.copy().astype(str))
            new_par_2_list = list(new_par_2)

            plt.xticks(new_par_1_list, new_par_1_ticks, fontsize=34)
            plt.yticks(new_par_2_list, new_par_2_ticks, fontsize=34)

            for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(34)

            ax.view_init(20,-60)

            plt.title("Acurácia da {}".format(labels[i]), fontsize=24, pad=30)

            plt.show()



    elif (len(keys) == 3) or (len(keys) == 4):
        
        if (len(keys) == 3):

            a, b, c = np.meshgrid(values[0], values[1], values[2])

            a = a.ravel() 
            b = b.ravel()
            c = c.ravel()

            data = [b,a,c]
        
        elif (len(keys) == 4):

            a, b, c, d = np.meshgrid(values[0], values[1], values[2], values[3])

            a = a.ravel() 
            b = b.ravel()
            c = c.ravel()
            d = d.ravel()

            data = [b,a,c,d]

        accuracy = accuracy.ravel()

        data.append(accuracy)

        data = np.array(data)

        data = data.T

        data = data[np.argsort(data[:,-1])]

        if colors == None:

            cmap_viridis = cm.get_cmap('viridis',len(keys)+1)
            colors_2 = cmap_viridis.colors

        else:

            colors_2 = colors

        cmap_rainbow = cm.get_cmap(cmap)

        colors_1 = cmap_rainbow(data.T[-1]/100)
        
        fig = plt.figure()

        axes = []
        x_pos = []

        for i in range(len(keys)):
            
            
            dx = 0.78/(len(keys))
            dy = 0.8

            ax = fig.add_axes([0.02 + i*dx, 0.1, dx, dy])

            ax.xaxis.set_visible(False)

            ax.set_xlim([0.02, 0.8])

            ax.set_ylim([np.min(dictionary[keys[i]]), np.max(dictionary[keys[i]])])

            locations = list(np.linspace(np.min(dictionary[keys[i]]), np.max(dictionary[keys[i]]), num=(len(values[i]))))

            ax.set_yticks(locations)

            ticks = ["" for j in range(len(values[i]))]
            
            ax.set_yticklabels(ticks)

            ax.annotate(str(np.min(dictionary[keys[i]])), (0.02 + i*dx, 0.07), xycoords='figure fraction', fontsize=30)

            ax.annotate(str(np.max(dictionary[keys[i]])), (0.02 + i*dx, 0.93), xycoords='figure fraction', fontsize=30)
            
            axes.append(ax)    

            if i < len(keys)-1:

                plt.grid(True)
                
                x = np.zeros(accuracy.size)+0.02

                x_pos.append(0.02)

                ax.plot(x, data.T[i],'.',color=colors_2[i],ms=10)

                ax.annotate(dic_names[keys[i]], (0.02 + i*dx, 0.035), xycoords='figure fraction', fontsize=30)

            if i == len(keys)-1:
                
                x = np.zeros(accuracy.size)+0.02

                x_pos.append(0.02)

                ax.plot(x, data.T[i],'.',color=colors_2[i],ms=10)

                ax.annotate(dic_names[keys[i]], (0.02 + i*dx, 0.035), xycoords='figure fraction', fontsize=30)

                ax_twin = ax.twinx()

                ax_twin.xaxis.set_visible(False)

                ax_twin.set_ylim([0, 100])

                locations = list(np.linspace(0, 100, num=(5)))

                ax_twin.set_yticks(locations)

                ticks = ["", "", "", "", "", ""]

                ax_twin.set_yticklabels(ticks)

                ax_twin.annotate(str(0), (0.02 + (i+1)*dx, 0.07), xycoords='figure fraction', fontsize=30)

                ax_twin.annotate(str(100), (0.02 + (i+1)*dx, 0.93), xycoords='figure fraction', fontsize=30)
                
                ax_twin.annotate("Acurácia (\%)", (0.02 + (i+1)*dx, 0.035), xycoords='figure fraction', fontsize=30)

                axes.append(ax_twin)

                x = np.zeros(accuracy.size)+0.8

                x_pos.append(0.8)

                ax_twin.plot(x, accuracy,'.',color=colors_2[i+1], ms=10)

                plt.grid(True)

        size = 0

        for i in range(len(a)):

            
            for j in range(len(x_pos)-1):

                xy1 = (x_pos[j],data[i,j])
                xy2 = (x_pos[j+1],data[i,j+1])
                connect = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA='data', coordsB='data',\
                    axesA=axes[j], axesB=axes[j+1], color=colors_1[i])
            
                if j < len(x_pos)-2:

                    axes[j].add_artist(connect)

                if j == len(x_pos)-2:

                    axes[j+1].add_artist(connect)

            if i >= len(a) - most_high - 1:
                

                xy = (1.2, 1.0 - size)
                
                if len(x_pos) - 2 == 3:
                    plt.annotate("{} - {}, {}, {}, {} - {} \%".format(str(i - (len(a) - most_high - 1)),data[i,0],data[i,1],data[i,2],data[i,3],\
                        np.around(data[i,4],decimals=2)), xy, xycoords='axes fraction', va='center', fontsize=20)
                elif len(x_pos) - 2 == 2:
                        plt.annotate("{} - {}, {}, {} - {} \%".format(str(i - (len(a) - most_high - 1)),data[i,0],data[i,1],data[i,2],\
                        np.around(data[i,3],decimals=2)), xy, xycoords='axes fraction', va='center', fontsize=20)


                size = size + 0.05

        
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

def trajectories(data, time, colors = None, cmap = 'viridis'):

    """ 
    .. py::function::

    This functions plots the neurons trajectories.

    :param data: the membrane potential data.
    :type data: numnpy.ndarray
    :param time: the time of the membrane potential.
    :type time: nupy.ndarray
    :param colors: a list of colorsa for every trajectory. Default is None.
    :type colors: list
    :param cmap: the cmap to use instead of a list of colors.
    :type cmap: str 
    
    """

    if ((not isinstance(data,np.ndarray)) and (not isinstance(time,np.ndarray))):

        raise TypeError("Data and Time must be np.ndarray.")

    if not isinstance(cmap,str):

        raise TypeError("Cmap must be of string type")

    if isinstance(colors, list) or isinstance(colors, np.ndarray):

        cols = colors

    else:
        color = cm.get_cmap(cmap, data.shape[1])
        
        if isinstance(color,mpl.colors.ListedColormap):

            cols = color.colors

        elif isinstance(color, mpl.colors.LinearSegmentedColormap):

            cols = color(range(data.shape[1].size))

        

        #color = cmap_trajecs(np.arange(data.shape[0])/data.shape[0])

    
    for i in range(data.shape[1]):

        plt.plot(time, data[:,i], color=cols[i])

    plt.xlabel("t [ms]", fontsize=34, labelpad=25)
    plt.ylabel("V [mV]", fontsize=34, labelpad=25)

    plt.xticks(fontsize=34)
    plt.yticks(fontsize=34)

    plt.grid(True)

    plt.show()

def compare_algs(accuracy_mean, cmap = 'viridis'):

    """ 
    .. py::function::

    This functions plots the algorithms comparinson accuracy.

    :param accuracy_mean: the accuracy mean with shape (3 x Number of Classes).
    :type accuracy_mean: numpy.ndarray
    :param accuracy_std: the accuracy standard devation with shape (3 x Number of Classes).
    :type accuracy_std: numpy.ndarray
    :param cmap: the cmap to use instead of a list of colors.
    :type cmap: str 
    
    """

    """ Create figure """
    fig = plt.figure(figsize=(10,10))

    """ Create Axis """
    ax = fig.gca(projection='3d')
   
    """ Set the Axis labels and Ticks """
    ax.set_xlabel("Nº de Sementes", fontsize=34, labelpad = 40)
    ax.set_ylabel("Métodos", fontsize=34, labelpad = 40)
    ax.set_zlabel("Classes", fontsize=34, labelpad = 40)

    ax.set_yticks([0,1,2])
    ax.set_yticklabels(['SCPL', 'LP', 'LS'], fontsize=34)

    arange = np.linspace(0,accuracy_mean.shape[2], num=6, dtype=int)
    ax.set_xticks(arange)
    ax.set_xticklabels(arange, fontsize=34)

    ax.set_zticks([i for i in range(accuracy_mean.shape[1])])
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(34)


    """ Set the colors """
    color_map = cm.get_cmap(cmap)
    color_mean = color_map(accuracy_mean/100)


    for i in range(accuracy_mean.shape[0]):
        for j in range(accuracy_mean.shape[1]):
            xs = np.arange(accuracy_mean.shape[2])
            ys = np.zeros(accuracy_mean.shape[2]) + j + 1
            if j == 0:
                ax.bar(xs, ys, zs = i, zdir='y', color = color_mean[i][j])
            elif j > 0:
                ax.bar(xs, ys-j, zs = i, zdir='y', color = color_mean[i][j], bottom=ys-1)
            

    normalize = mpl.colors.Normalize(vmin=0, vmax=100)
    
    sm = mpl.cm.ScalarMappable(norm=normalize, cmap = cmap)
    
    cbar = fig.colorbar(sm, ticks=np.linspace(0,100,11))
    cbar.ax.set_yticklabels(np.linspace(0,100,11).astype(int).astype(str), fontsize=30)
    cbar.ax.set_ylabel('Acurácia', fontsize=30)


    plt.show()

def diameters(classified, diameters, fit=True, data_col='b', fit_col='r', deg = 3):

    """ 
    .. py::function::

    This function plots the diameters values per amount of classified data points.

    :param classified: Amount of classified data points
    :type classified: numpy.ndarray
    :param diameters: Diameters values.
    :type diameters: numpy.ndarray
    :param fit: Define if will fit the data. Default is True.
    :type fit: bool.
    :param data_col: Color of the data points. Deafault is blue.
    :type data_col: str
    :param fit_col: Color of the fit. Deafault is red.
    :type fit_col: str
    :param deg: degree of the polynomial fitting. Default is 3
    :type deg: int

    """

    if isinstance(fit,bool) and fit==True and isinstance(deg,int):

        z = np.polyfit(classified, diameters, deg)
        function = np.poly1d(z)

        x_new = np.arange(classified[0],classified[-1])
        y_new = function(x_new)

        indexes = np.where(y_new>0)[0]
        if len(indexes) != 0:
            plt.plot(x_new[indexes],y_new[indexes],fit_col,label='Ajuste')
        else:
            plt.plot(x_new,y_new,fit_col,label='Ajuste')

    plt.scatter(classified, diameters, s=100, c=data_col, label="Dados")
    plt.xlabel("Nº Dados Classificados", fontsize=34, labelpad=30)
    plt.ylabel("D", fontsize=34, labelpad=30)
    plt.xticks(fontsize=34)
    plt.yticks(fontsize=34)
    plt.annotate("D = {} cm".format(round(diameters[-1],3)), (0.85,0.7), xycoords='figure fraction', fontsize=30)
    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=34)
    plt.show()

def diameters_per_seeds(seeds_array, diameters_array, show_value = True, stds = None, cmap = 'viridis', **kwargs):

    """ 
    .. py::function::

    This function calculates the diameters means and the standard deviations per seeds.

    :param seeds_array: The seeds array
    :type seeds_array: numpy.ndarray
    :param diameters_array: The diameters array
    :type diameters_array: numpy.ndarray

    :param show_value: Show the diameters per seed. Default is True.
    :type show_value: bool
    :param stds: How many standard deviations will be shown. Default None.
    :type stds: int.
    :param cmap: a string value that represents a colormap
    :type cmap: str
    
    :param \**kwargs:
        See below

    :Keyword Arguments:
        
        * *colors* (``list``) --
        A list of colors that represents each group synchronized.

        * *std_array* (``numpy.ndarray``)
        The array of standard deviations

    """

    if ((not isinstance(seeds_array,np.ndarray))and(not isinstance(diameters_array,np.ndarray))):

        raise TypeError("Seeds and diameters must be of the type numpy.ndarray.")

    if ((stds != None) and (not isinstance(stds, int))):

        raise TypeError("Stds must be of the type int.")

    colors = kwargs.get("colors")

    std_array = kwargs.get("std_array")

    if colors == None:

        color = cm.get_cmap(cmap,len(labels)).colors
    
    else:

        color = colors

    if stds != None:

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_xlabel(r'Nº Sementes', fontsize=36)
        ax.set_ylabel(r'D', fontsize=36)

        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)

        line = {"linestyle":"-", "linewidth":2, "markeredgewidth":2,\
        "elinewidth":2, "capsize":3}

      

        ax.errorbar(seeds_array, diameters_array, yerr = std_array*stds, \
            **line, color = color)

        if show_value == True:

            diameters_array = np.around(diameters_array, decimals=2)
            for i, txt in enumerate(diameters_array):   
                ax.annotate(txt, xy=(seeds_array[i], diameters_array[i]),
                xytext=(seeds_array[i]+0.03,diameters_array[i]+0.3),
                color=color) 

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=34)
        plt.grid(True)
        plt.show()

    elif stds == None:
        
        fig = plt.figure()
        ax = fig.add_subplot(111)


        ax.set_xlabel(r'Nº Sementes', fontsize=36)
        ax.set_ylabel(r'D', fontsize=36)

        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)

        #line = {"linestyle":"-", "linewidth":2, "markeredgewidth":2,\
        #"capsize":3}

        ax.plot(seeds_array, diameters_array, color = color)

        if show_value == True:

            diameters_array = np.around(diameters_array, decimals=2)
            for i, txt in enumerate(diameters_array[i]):   
                ax.annotate(txt, xy=(seeds_array[i], diameters_array[i]),
                xytext=(seeds_array[i]+0.03,diameters_array[i]+0.3),
                color=color) 

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=34)
        plt.grid(True)
        plt.show()

def coupling(data_diff, phase_diff, force, time, cmap='viridis'):


    """ 
    .. py:function:

    This function calculates the phases and trajectories difference by varying the coupling force.

    Args:

        :param data: Voltage difference between the oscillators. Shape: (Force vector, time vector) 
        :type data: np.ndarray
        :param phases: Phases Difference between the oscillators. Shape: (Force vector, time vector)
        :type phases: np.ndarray
        :param force: A vector of the coupling force variation.
        :type force: np.ndarray
        :param cmap: a string value that represents a colormap
        :type cmap: str

    """

    data = [data_diff, phase_diff]
    labels = [r'$|V_{i} - V_{j}|$', r'$|\phi_{i} - \phi_{j}|$']

    for i in range(2):
    
        fig, ax = plt.subplots(figsize=(10,10),subplot_kw=dict(projection='3d'))

        k, times = np.meshgrid(force, time)

        color = cm.get_cmap(cmap,force.size)

        if isinstance(color,mpl.colors.ListedColormap):

            cols = color.colors

        elif isinstance(color, mpl.colors.LinearSegmentedColormap):

            cols = color(range(force.size))
    
        """ Customize the X, Y and Z Ticks """
        new_force = np.around(np.linspace(force[0],force[-1],(6), dtype=float),decimals=2)
        force_ticks = list(new_force.copy().astype(str))
        force_list = list(new_force)

        new_time = np.linspace(0,time[-1],(5), dtype=int) + 1
        time_ticks = list(new_time.copy().astype(str))
        time_list = list(new_time)
        
        surface = ax.plot_surface(k.T, times.T, data[i], cmap=cmap, edgecolor=None)
        cbar = fig.colorbar(surface, shrink=0.5,aspect=5, pad=0.1, ax=ax)
        cbar.ax.tick_params(labelsize=24)

        """ Set x, y, and z """
        ax.set_xlabel('Força de Acoplamento (k)', fontsize=34, labelpad=40)
        ax.set_ylabel('t', fontsize=30, labelpad=40)
        ax.set_zlabel(labels[i],fontsize=34, labelpad=40)

        """ Customize the X, Y and Z Ticks """
        plt.yticks(time_list, time_ticks, fontsize=34)
        plt.xticks(force_list, force_ticks, fontsize=34)
        for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(34)

        """ Customize the angle """
        ax.view_init(20,-60)

        """ Show the Plot """
        plt.show()

def periods_mean(periods):

    means = np.array([])
    stds = np.array([])
    intervals = np.array([])
    for i in range(len(periods)):

        period = np.array(periods[i])
        mean = np.mean(period,axis=0)
        std = np.std(period,axis=0)

        means = np.append(means,mean)
        stds = np.append(stds,std)
        intervals = np.append(intervals,len(periods[i]))

    interval_mean = np.mean(intervals)
    interval_std = np.std(intervals)

    """ Define them as strings """
    mean_str = str(round(interval_mean,2))
    std_str = str(round(interval_std,2))

    """ Define the amount of bins by subtracting the highest value by the minimum """
    bins = int(intervals.max() - intervals.min())
    if bins < 10:
        bins = 10

    """ Generate the histogram array, and the bins array """
    hist, bins_edges = np.histogram(intervals,bins)

    """ Generate X values between the minimum and maximum values of the interval """
    x = np.linspace(intervals.min(),intervals.max(),(1000))

    """ Calculate the normal distribution by inserting the x array"""
    distribution = ((1 / (np.sqrt(2 * np.pi) * interval_std)) *\
        np.exp(-0.5 * (1 / interval_std * (x - interval_mean))**2))

    """ Create the frequency color by dividing the histogram array by it's highest value """
    freq_color = hist/hist.max()

        
    """ Create the colors array """
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(freq_color.min(), freq_color.max())
    colors = cmap(freq_color)


    """ Create the figure """
    fig = plt.figure()

    """ Create the axe for the frequency/histogram (ax1),
        the axe for the normal distribution (ax2), and the
        ax for the colorbar (ax3) """
    ax1 = fig.add_axes([0.05, 0.3, 0.9, 0.7]) 
    ax2 = ax1.twinx()                         
    ax3 = fig.add_axes([0.05, 0.1, 0.9, 0.1])

    """ Create the color bar at the horizontal """
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, norm=norm,\
        orientation='horizontal')

    """ Calculate the bins width """
    if len(bins_edges)>2:
        width = bins_edges[2]-bins_edges[1]
    else:
        width = bins_edges[1]-bins_edges[0]

    """ Font configuration """
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    """ Plot the bars """
    ax1.bar(bins_edges[:-1], hist, width, color=colors, edgecolor ='k', alpha=0.8)

    """ Configuration of the labels and ticks of the horizontal color bar  """
    cb1.ax.tick_params(labelsize=24)
    cb1.ax.set_xlabel(r'\textbf{Frequência Normalizada} ($\mathbf{f/f_{max}}$)', fontsize=24)

    """ Configuration of the bar plot (ax1) """
    ax1.set_xlabel(r'\textbf{Número de Intervalos}', fontsize=24)
    ax1.set_ylabel(r'\textbf{Frequência (f)}',fontsize=24)
    ax1.tick_params(axis='x', labelsize=24)
    ax1.tick_params(axis='y', labelsize=24)

    """ Normal Distribution plot and axe configuration """
    ax2.plot(x, distribution,'r--', label = r'$\mathbf{\mu = %s\, \sigma = %s}$'%(mean_str,std_str))
    ax2.legend(loc='best', fontsize=24)
    ax2.tick_params(axis='y',labelsize=24)
    ax2.set_ylabel(r'\textbf{Densidade de Probabilidade}', fontsize=24)

    """ Show the Graph """
    plt.show()

        
    """" Obtain the Mean and the Standard Deviation of the Means"""
    total_means = np.mean(means)
    total_stds = np.std(stds)

    """ Define them as strings """
    means_str = str(round(total_means,2))
    stds_str = str(round(total_stds,2))

    """ Define an array of bins between the minimum and maximum values of means with 12 bins """
    #bins = np.linspace(means.min(), means.max(), num=12)
    bins = int(means.max() - means.min())
    if bins < 12:
        bins = 12

    """ Generate the histogram array, and the bins array """
    hist, bins_edges = np.histogram(means,bins)

    """ Generate X values between the minimum and maximum values of the means """
    x = np.linspace(means.min(),means.max(),(1000))

    """ Calculate the normal distribution by inserting the x array"""
    distribution = ((1 / (np.sqrt(2 * np.pi) * total_stds)) *\
        np.exp(-0.5 * (1 / total_stds * (x - total_means))**2))

    """ Create the frequency color by dividing the histogram array by it's highest value """
    freq_color = hist/hist.max()

    """ Create the colors array """
    cmap = plt.cm.plasma
    norm = mpl.colors.Normalize(freq_color.min(), freq_color.max())
    colors = cmap(freq_color)


    """ Create the figure """
    fig = plt.figure()

    """ Create the axe for the frequency/histogram (ax1),
        the axe for the normal distribution (ax2), and the
        ax for the colorbar (ax3) """
    ax1 = fig.add_axes([0.05, 0.3, 0.9, 0.7]) 
    ax2 = ax1.twinx()                         
    ax3 = fig.add_axes([0.05, 0.1, 0.9, 0.1])

    """ Create the color bar at the horizontal """
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, norm=norm,\
        orientation='horizontal')

    """ Calculate the bins width """
    if len(bins_edges)>2:
        width = bins_edges[2]-bins_edges[1]
    else:
        width = bins_edges[1]-bins_edges[0]

    """ Font configuration """
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    """ Plot the bars """
    ax1.bar(bins_edges[:-1], hist, width, color=colors, edgecolor ='k', alpha=0.8)

    """ Configuration of the labels and ticks of the horizontal color bar  """
    cb1.ax.tick_params(labelsize=24)
    cb1.ax.set_xlabel(r'\textbf{Frequência Normalizada} ($\mathbf{f/f_{max}}$)', fontsize=24)

    """ Configuration of the bar plot (ax1) """
    ax1.set_xlabel(r'\textbf{Médias dos Períodos (T)}', fontsize=24)
    ax1.set_ylabel(r'\textbf{Frequência (f)}',fontsize=24, labelpad=-5)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)

    """ Normal Distribution plot and axe configuration """
    ax2.plot(x, distribution,'r--', label = r'$\mathbf{\mu = %s\, \sigma = %s}$'%(means_str,stds_str))
    ax2.legend(loc='best', fontsize=24)
    ax2.tick_params(axis='y',labelsize=20)
    ax2.set_ylabel(r'\textbf{Densidade de Probabilidade}', fontsize=24)

    """ Show the Graph """
    plt.show()



