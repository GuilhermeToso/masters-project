import numpy as np
import pandas as pd
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib as mpl
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
import inspect
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams['text.usetex'] = True

""" Plotar Pesos """
def plot_weight(colormaped = None, color = 'r', cmap='viridis',**kwargs):


    keys = [key for key in kwargs.keys()]
    values = [val for val in kwargs.values()]
    xlabel = {'weight':r'w','step':r'$\Delta w$', 'beta':r'$\beta$', 'gamma':r'$\gamma$',
            'degree_in':r'$Grau\hspace{0.1cm}Interno\hspace{0.1cm}(G^{in})$', 'degree_coming':r'$Grau\hspace{0.1cm}Convergente\hspace{0.1cm}(G_{conv})$'}
    w_label = r'$K_{ij}$'
    r_label = r'Reforço ($K_{ij}^{r}$)'
    p_label = r'Punição ($K_{ij}^{p}$)'
    title = "Conexão entre os neurônios i e j pertencentes ao grupo 1,\n e punidos pelo grupo 2"
    x_key = None
    for i in range(len(keys)):
        if isinstance(values[i],np.ndarray) and keys[i] != colormaped:
            x_key = keys[i]
            break

    colormap = cm.get_cmap(cmap,kwargs[colormaped].size)

    if isinstance(colormap,mpl.colors.ListedColormap):
        colors = colormap.colors
    elif isinstance(colormap,mpl.colors.LinearSegmentedColormap):
        colors = colormap(range(kwargs[colormaped].size))

    fig = plt.figure()
    ax = fig.add_subplot(121)

    initial_weight = kwargs['weight'] + np.zeros(shape=(kwargs[x_key].size))
 
    if colormaped == 'beta':

        for i in range(kwargs[colormaped].size):

            punish = punishment(kwargs['step'], kwargs['beta'][i], kwargs['degree_coming']) + np.zeros(shape=(kwargs[x_key].size))
            ax.scatter(kwargs[x_key], punish, s=100, c = colors[i])
        reinforce = reinforcement(kwargs['step'],kwargs['degree_coming'], kwargs['degree_in'], kwargs['gamma']) + np.zeros(shape=(kwargs[x_key].size))
        ax.scatter(kwargs[x_key], reinforce, s=100, c = 'r', label='Reforço')
        plt.grid()
        plt.xlabel(xlabel[x_key], fontsize =30, labelpad=30)
        plt.ylabel(p_label, fontsize =30, labelpad=30)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.suptitle(title, fontsize=20)
        plt.legend()

        ax1 = fig.add_subplot(122)
        for i in range(kwargs[colormaped].size):
            
            punish = punishment(kwargs['step'], kwargs['beta'][i], kwargs['degree_coming']) + np.zeros(shape=(kwargs[x_key].size))
            reinforce = reinforcement(kwargs['step'],kwargs['degree_coming'], kwargs['degree_in'], kwargs['gamma']) + np.zeros(shape=(kwargs[x_key].size))
            weight = update_weight(initial_weight, reinforce, punish)
            ax1.plot(kwargs[x_key], weight, c=colors[i])
        ax1.plot(kwargs[x_key], initial_weight, c='m',label='Peso Inicial')
        normalize = mpl.colors.Normalize(vmin=kwargs[colormaped].min(), vmax=kwargs[colormaped].max())
        sm = mpl.cm.ScalarMappable(norm=normalize, cmap = cmap)
        cbar = fig.colorbar(sm, ticks=np.linspace(kwargs[colormaped].min(),kwargs[colormaped].max(),11))
        cbar.ax.set_yticklabels(np.round(np.linspace(kwargs[colormaped].min(),kwargs[colormaped].max(),11),decimals=2).astype(str), fontsize=30)
        cbar.ax.set_ylabel(r'$\beta$', fontsize=30)
        plt.grid()
        plt.xlabel(xlabel[x_key], fontsize =30, labelpad=30)
        plt.ylabel(w_label, fontsize =30, labelpad=15)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.suptitle(title, fontsize=20)
        plt.legend()
        plt.show()


        
    elif colormaped == 'gamma':

        for i in range(kwargs[colormaped].size):
            reinforce = reinforcement(kwargs['step'],kwargs['degree_coming'], kwargs['degree_in'], kwargs['gamma'][i]) + np.zeros(shape=(kwargs[x_key].size))
            ax.scatter(kwargs[x_key], reinforce, s=100, c = colors[i])
        punish = punishment(kwargs['step'], kwargs['beta'], kwargs['degree_coming']) + np.zeros(shape=(kwargs[x_key].size))
        ax.scatter(kwargs[x_key], punish, s=100, c = 'r', label='Punição')
        plt.grid()
        plt.xlabel(xlabel[x_key], fontsize =30, labelpad=30)
        plt.ylabel(r_label, fontsize =30, labelpad=30)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.suptitle(title, fontsize=20)
        plt.legend()

        
        ax1 = fig.add_subplot(122)
        for i in range(kwargs[colormaped].size):
            
            punish = punishment(kwargs['step'], kwargs['beta'], kwargs['degree_coming']) + np.zeros(shape=(kwargs[x_key].size))
            reinforce = reinforcement(kwargs['step'],kwargs['degree_coming'], kwargs['degree_in'], kwargs['gamma'][i]) + np.zeros(shape=(kwargs[x_key].size))
            weight = update_weight(initial_weight, reinforce, punish)
            ax1.plot(kwargs[x_key], weight, c=colors[i])
        ax1.plot(kwargs[x_key], initial_weight, c='m',label='Peso Inicial')
        normalize = mpl.colors.Normalize(vmin=kwargs[colormaped].min(), vmax=kwargs[colormaped].max())
        sm = mpl.cm.ScalarMappable(norm=normalize, cmap = cmap)
        cbar = fig.colorbar(sm, ticks=np.linspace(kwargs[colormaped].min(),kwargs[colormaped].max(),11))
        cbar.ax.set_yticklabels(np.round(np.linspace(kwargs[colormaped].min(),kwargs[colormaped].max(),11),decimals=2).astype(str), fontsize=30)
        cbar.ax.set_ylabel(r'$\gamma$', fontsize=30)
        plt.grid()
        plt.xlabel(xlabel[x_key], fontsize =30, labelpad=30)
        plt.ylabel(w_label, fontsize =30, labelpad=15)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.suptitle(title, fontsize=20)
        plt.legend()
        plt.show()
        

    elif colormaped == 'degree_in':

        for i in range(kwargs[colormaped].size):
            reinforce = reinforcement(kwargs['step'],kwargs['degree_coming'], kwargs['degree_in'][i], kwargs['gamma']) + np.zeros(shape=(kwargs[x_key].size))
            ax.scatter(kwargs[x_key], reinforce, s=100, c = colors[i])
        punish = punishment(kwargs['step'], kwargs['beta'], kwargs['degree_coming']) + np.zeros(shape=(kwargs[x_key].size))
        ax.scatter(kwargs[x_key], punish, s=100, c = 'r', label='Punição')
        plt.grid()
        plt.xlabel(xlabel[x_key], fontsize =30, labelpad=30)
        plt.ylabel(r_label, fontsize =30, labelpad=30)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.suptitle(title, fontsize=20)
        plt.legend()

        
        ax1 = fig.add_subplot(122)
        for i in range(kwargs[colormaped].size):
            
            punish = punishment(kwargs['step'], kwargs['beta'], kwargs['degree_coming']) + np.zeros(shape=(kwargs[x_key].size))
            reinforce = reinforcement(kwargs['step'],kwargs['degree_coming'], kwargs['degree_in'][i], kwargs['gamma']) + np.zeros(shape=(kwargs[x_key].size))
            weight = update_weight(initial_weight, reinforce, punish)
            ax1.plot(kwargs[x_key], weight, c=colors[i])
        ax1.plot(kwargs[x_key], initial_weight, c='m',label='Peso Inicial')
        normalize = mpl.colors.Normalize(vmin=kwargs[colormaped].min(), vmax=kwargs[colormaped].max())
        sm = mpl.cm.ScalarMappable(norm=normalize, cmap = cmap)
        cbar = fig.colorbar(sm, ticks=np.linspace(kwargs[colormaped].min(),kwargs[colormaped].max(),11))
        cbar.ax.set_yticklabels(np.round(np.linspace(kwargs[colormaped].min(),kwargs[colormaped].max(),11),decimals=2).astype(str), fontsize=30)
        cbar.ax.set_ylabel(xlabel[colormaped], fontsize=30)
        plt.grid()
        plt.xlabel(xlabel[x_key], fontsize =30, labelpad=30)
        plt.ylabel(w_label, fontsize =30, labelpad=15)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.suptitle(title, fontsize=20)
        plt.legend()
        plt.show()


def plot_punish_reinforce_condition(degree_in,degree_incident,beta,gamma,cmap=cm.viridis):

    xx,yy = np.meshgrid(degree_in,degree_incident)
    zz = beta/xx - np.exp(-gamma*yy)/yy


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surface = ax.plot_surface(xx,yy,zz,cmap=cmap,linewidth=0, antialiased=False)

    z = list(np.linspace(zz.min(),zz.max(),num=5))
    x_ticks = np.linspace(degree_in.min(),degree_in.max(),num=5)
    ax.set_zticks(z)

    ax.set_xlabel(r'$G^{in}$', fontsize=30, labelpad=60)
    ax.set_ylabel(r'$G^{-}$', fontsize=30, labelpad=60)
    ax.set_zlabel(r'$\dfrac{\beta}{G^{in}} - \dfrac{e^{-\gamma G^{-}}}{G^{-}}$',fontsize=34, labelpad=60,rotation=90)
    ax.set_xticks(x_ticks)
    plt.xticks(fontsize=30)

    plt.yticks(fontsize=30)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(30)
    cb = fig.colorbar(surface, shrink=0.5, aspect=5,pad=0.1)
    cb.ax.tick_params(labelsize=30)
    plt.show()

def plot_sigmoid(function, x, colormaped, colormaped_name, cmap='viridis'):

    colormap = cm.get_cmap(cmap,colormaped.size)

    if isinstance(colormap,mpl.colors.ListedColormap):
        color = colormap.colors
    elif isinstance(colormap,mpl.colors.LinearSegmentedColormap):
        color = colormap(range(colormaped.size))


    fig, ax = plt.subplots()
    normalize = mpl.colors.Normalize(vmin=colormaped.min(),vmax=colormaped.max())
    sm = mpl.cm.ScalarMappable(norm=normalize,cmap=cmap)
    cbar = fig.colorbar(sm,ticks=np.linspace(colormaped.min(),colormaped.max(),8))
    cbar.ax.set_yticklabels(np.around(np.linspace(colormaped.min(),colormaped.max(),8),decimals=3).astype(str), fontsize=30)
    cbar.ax.set_ylabel(colormaped_name, fontsize=30)

    for i in range(colormaped.size):
        ax.plot(x,function[i],c=color[i])
    plt.grid()
    plt.xlabel('x',fontsize=34, labelpad=30)
    plt.ylabel('f(x)',fontsize=34, labelpad=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.show()

def plot_reinforce_term(colormaped = None, cmap='viridis', **kwargs):


    keys = [key for key in kwargs.keys()]
    values = [val for val in kwargs.values()]
    
    xlabel = {'degree_in':r'$G^{in}$', 'degree_coming':r'$G^{-}$', 'gamma':r'$\gamma$'}
    ylabel = r'$e^{-\gamma G^{-}}G^{in}$'

    x_key = None
    for i in range(len(keys)):
        if isinstance(values[i],np.ndarray) and keys[i] != colormaped:
            x_key = keys[i]
            break

    colormap = cm.get_cmap(cmap,kwargs[colormaped].size)

    if isinstance(colormap,mpl.colors.ListedColormap):
        colors = colormap.colors
    elif isinstance(colormap,mpl.colors.LinearSegmentedColormap):
        colors = colormap(range(kwargs[colormaped].size))

    def function(degree_in,degree_coming,gamma):
        r_term = np.exp(-gamma*degree_coming)*degree_in
        return r_term

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if colormaped=='degree_in':

        for i in range(kwargs[colormaped].size):        
            plt.plot(kwargs[x_key], function(kwargs['degree_in'][i], kwargs['degree_coming'], kwargs['gamma']),c=colors[i])
            plt.plot(kwargs[x_key],np.zeros(shape=(kwargs[x_key].size))+6,c='r')
        normalize = mpl.colors.Normalize(vmin=kwargs[colormaped].min(), vmax=kwargs[colormaped].max())
        sm = mpl.cm.ScalarMappable(norm=normalize, cmap = cmap)
        cbar = fig.colorbar(sm, ticks=np.linspace(kwargs[colormaped].min(),kwargs[colormaped].max(),11))
        cbar.ax.set_yticklabels(np.round(np.linspace(kwargs[colormaped].min(),kwargs[colormaped].max(),11),decimals=2).astype(str), fontsize=30)
        cbar.ax.set_ylabel(xlabel[colormaped], fontsize=30)        
        plt.grid()
        plt.xlabel(xlabel[x_key], fontsize =30, labelpad=30)
        plt.ylabel(ylabel, fontsize =30, labelpad=30)
        plt.ylim(0,20)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.show() 

    if colormaped=='degree_coming':

        for i in range(kwargs[colormaped].size):        
            plt.plot(kwargs[x_key], function(kwargs['degree_in'], kwargs['degree_coming'][i], kwargs['gamma']),c=colors[i])
            plt.plot(kwargs[x_key],np.zeros(shape=(kwargs[x_key].size))+6,c='r')
        normalize = mpl.colors.Normalize(vmin=kwargs[colormaped].min(), vmax=kwargs[colormaped].max())
        sm = mpl.cm.ScalarMappable(norm=normalize, cmap = cmap)
        cbar = fig.colorbar(sm, ticks=np.linspace(kwargs[colormaped].min(),kwargs[colormaped].max(),11))
        cbar.ax.set_yticklabels(np.round(np.linspace(kwargs[colormaped].min(),kwargs[colormaped].max(),11),decimals=2).astype(str), fontsize=30)
        cbar.ax.set_ylabel(xlabel[colormaped], fontsize=30)        
        plt.grid()
        plt.xlabel(xlabel[x_key], fontsize =30, labelpad=30)
        plt.ylabel(ylabel, fontsize =30, labelpad=30)
        plt.ylim(0,20)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.show() 

    if colormaped=='gamma':

        for i in range(kwargs[colormaped].size):        
            plt.plot(kwargs[x_key], function(kwargs['degree_in'], kwargs['degree_coming'], kwargs['gamma'][i]),c=colors[i])
            plt.plot(kwargs[x_key],np.zeros(shape=(kwargs[x_key].size))+6,c='r')
        normalize = mpl.colors.Normalize(vmin=kwargs[colormaped].min(), vmax=kwargs[colormaped].max())
        sm = mpl.cm.ScalarMappable(norm=normalize, cmap = cmap)
        cbar = fig.colorbar(sm, ticks=np.linspace(kwargs[colormaped].min(),kwargs[colormaped].max(),11))
        cbar.ax.set_yticklabels(np.round(np.linspace(kwargs[colormaped].min(),kwargs[colormaped].max(),11),decimals=2).astype(str), fontsize=30)
        cbar.ax.set_ylabel(xlabel[colormaped], fontsize=30)        
        plt.grid()
        plt.xlabel(xlabel[x_key], fontsize =30, labelpad=30)
        plt.ylabel(ylabel, fontsize =30, labelpad=30)
        plt.ylim(0,20)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.show()   

def plot_exp(x, function):

    plt.plot(x,function,c='b')
    plt.grid()
    plt.xlabel('x',fontsize=34, labelpad=30)
    plt.ylabel(r'$1-e^{-x}$',fontsize=34, labelpad=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.show()

""" Plot exponential """
def exponential(x):
    return np.exp(-x)


""" Weight Reinforcement Term """
def reinforcement(step, degree_coming, degree_in, gamma):

    return step*(1-np.exp(-degree_in*np.exp(-gamma*degree_coming)))

""" Weight Punishment Term """
def punishment(step, beta, degree_coming):

    return step*(1-np.exp(-beta*degree_coming))

""" Weight Update Function """
def update_weight(weight, reinforce, punish):

    weight = weight + reinforce - punish

    return weight

""" Initial Weight """
w_max = 1.0
w_min = 0.1
w = w_min + (w_max - w_min)*0.5

step = 0.3 # Maximum and Minimum weight update value
beta =np.linspace(0.001,2,num=100)
gamma = 0.3#np.linspace(0.1,3,num=100)#0.3 # Punishment strengh term
degree_in = 1000#np.linspace(100,1000,num=100)#1000## Internal degree of the group i
degree_coming = np.arange(50) # External degree coming from groups j, j+1, ..., K to group i


plot_weight(colormaped='beta', weight=w,step=step,degree_in=degree_in, degree_coming=degree_coming, beta=beta,gamma=gamma)

#plot_reinforce_term(colormaped='degree_coming',degree_in=degree_in, degree_coming=degree_coming,gamma=gamma)

#plot_punish_reinforce_condition(degree_in, degree_coming, beta, gamma)
#plot_exp(x,1-exponential(x))



#plot_weight(colormaped='degree_in', weight=weight, step=step, alpha=alpha, beta=beta, degree_in=degree_in, degree_coming=degree_coming)

