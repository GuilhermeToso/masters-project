# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 22:22:01 2020

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import opencv


clusters = np.load('cluster_seeds.npy')
clusters_std = np.load('cluster_seeds_std.npy')


circles = np.load('circles_seeds.npy')
circles_std = np.load('circles_seeds_std.npy')

moons = np.load('moons_seeds.npy')
moons_std = np.load('moons_seeds_std.npy')

classification = np.load('classification_seeds.npy')
classification_std = np.load('classification_seeds_std.npy')

gaussian = np.load('gaussian_seeds.npy')
gaussian_std = np.load('gaussian_seeds_std.npy')

print(gaussian)