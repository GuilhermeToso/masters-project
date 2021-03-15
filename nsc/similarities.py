import numpy as np
import scipy.spatial.distance as dist
import sys

__all__ = ['euclidean','hamming','manhattan','jaccard','cosine','imed']

def euclidean(x,y,axis):

    """ 
    .. py::function:

    This function calculates the euclidean distance as the similarity between two nodes in a graph.

    param x: the first node.
    type x: np.ndarray
    param y: the second node.
    type y: np.ndarray
    param axis: the axis in which will be realized the operation.
    type axis: int

    """
    
    return np.linalg.norm(x-y,axis=axis)

def hamming(x,y, axis):

    return (x.size - np.sum(x==y, axis=1))/x.size

def manhattan(x,y, axis):

    return dist.cityblock(x,y)

def jaccard(x,y, axis):

    return dist.jaccard(x,y)

def cosine(x,y, axis):

    return dist.cosine(x,y)

def imed(x,y,axis):

    r = 2

    img_1 = x
    img_2 = y
    
    rows_1, cols_1 = np.where(img_1==img_1)
    rows_2, cols_2 = np.where(img_2==img_2)

    pos_row_1 = rows_1[None,:]
    pos_col_1 = cols_1[None,:]
    
    pos_row_2 = rows_2[:,None]
    pos_col_2 = cols_2[:,None]

    pixel_dist = (pos_row_2-pos_row_1)**2 + (pos_col_2-pos_col_1)**2
    G = (1/2*np.pi*r**2)*np.exp(-pixel_dist/2*r**2)
    
    img_1_flat = img_1.flatten()
    img_2_flat = img_2.flatten()
    
    distance = dist.mahalanobis(img_1_flat, img_2_flat, G)
    
    return distance
