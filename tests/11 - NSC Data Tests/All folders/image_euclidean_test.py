# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 12:03:54 2020

@author: User
"""

import numpy as np
import sys
import scipy

s1,s2=5,5

#np.random.seed(0)
array = np.zeros(shape=(10000,))
for i in range(array.size):
    img_1 = np.random.randint(0,255,size=(s1,s2))
    img_2 = np.random.randint(0,255,size=(s1,s2))
    
    rows_1, cols_1 = np.where(img_1==img_1)
    rows_2, cols_2 = np.where(img_2==img_2)
    
    pos_row_1 = rows_1[None,:]
    pos_col_1 = cols_1[None,:]
    
    pos_row_2 = rows_2[:,None]
    pos_col_2 = cols_2[:,None]
    
    dist = (pos_row_1-pos_row_2)**2 + (pos_col_1-pos_col_2)**2
    
    r = 2
    G = np.exp(-dist/2*r**2)
    
    x = img_1.flatten()
    y = img_2.flatten()
    
    d = scipy.spatial.distance.mahalanobis(x,y,G)
    array[i] = d

print("Max: ", array.max())
print("Min: ", array.min())    


sys.exit(0)


img_1_row = img_1.flatten()[None,:]
img_2_col = img_2.flatten()[:,None]
print("Img 1 as row: \n {}".format(img_1_row))
print("Img 2 as col: \n {}".format(img_2_col))
print("Pixels Diff: \n {}".format(img_1_row - img_2_col))


img_1_col = img_1.flatten()[:,None]
img_2_row = img_2.flatten()[None,:]
print("Img 1 as col: \n {}".format(img_1_col))
print("Img 2 as row: \n {}".format(img_2_row))
print("Pixels Diff: \n {}".format(img_1_col - img_2_row))
#
#img_1_r = img_1.flatten()
#img_2_c = img_2.flatten()
#print("Img 1 Pixels Flat: \n{}".format(img_1_r))
#print("Img 2 Pixels Flat: \n{}".format(img_2_c))
diff = img_1_col.T - img_2_row
print("Difference: \n{}".format(diff))
print(diff + np.zeros(diff.shape[1])[:,None])
a = diff + np.zeros(diff.shape[1])[:,None]

d = np.sum(a*G)/2*np.pi
print(d)
def imed(img_1, img_2, r):


    rows_1, cols_1 = np.where(img_1==img_1)
    rows_2, cols_2 = np.where(img_2==img_2)
    
    pos_row_1 = rows_1[None,:]
    pos_col_1 = cols_1[None,:]
    
    pos_row_2 = rows_2[:,None]
    pos_col_2 = cols_2[:,None]
    
    dist = (pos_row_2-pos_row_1)**2 + (pos_col_2-pos_col_1)**2
    
    G = (1/2*np.pi*r**2)*np.exp(-dist/2*r**2)
    
    img_1_row = img_1.flatten()[None,:]
    img_2_col = img_2.flatten()[:,None]
    
    
    pixels_diff = img_1_row - img_2_col
    pixels_diff_transpose = pixels_diff.T
    return np.sqrt(np.sum(pixels_diff*G*pixels_diff_transpose))    
# =============================================================================
# 
# for i in range(100):
#         
#     img_1 = np.random.randint(0,20,size=(50,50))
#     img_2 = np.random.randint(0,20,size=(50,50))
#     
#     a = imed(img_1,img_2,2)
#     print(a)
# =============================================================================
