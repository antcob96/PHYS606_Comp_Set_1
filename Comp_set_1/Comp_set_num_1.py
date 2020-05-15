# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:06:55 2020

@author: tonyc
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.cbook as cbook
import math as mth
import scipy.signal as sy
import random as random
from matplotlib.patches import PathPatch

from mpl_toolkits import mplot3d

import pandas as pd
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def random_location(squareSize):
    L = squareSize 
    
    val = random.randint(1, 4) 
    ran0 = random.random()
    if val == 1: 
        location[0] = (ran0)*L
        location[1] = 0 
    elif val == 2: 
        location[0] = (ran0)*L
        location[1]  = L 
    elif val == 3: 
        location[0] = 0
        location[1]  = (ran0)*L 
    else: 
        location[0] = L
        location[1]  = (ran0)*L 
    return location



L_size = 10
# Step size  is a 
M = np.zeros((10,10))
n_steps = 1000
location = np.zeros((2,))

location = random_location(L_size)
i = np.int(location[0])-1
j = np.int(location[1])-1
M[i,j] = M[i,j] + 1

for n in range(1,n_steps):
    val = random.randint(1, 4) 
    if val == 1: 
        i = i + 1
        j = j 
    elif val == 2: 
        i = i
        j = j + 1
    elif val == 3: 
        i = i
        j = j - 1
    else: 
        i = i - 1
        j = j 
    if (i == 10):
        i = i - 1
    elif (j == 10):
        j = j - 1
    elif (i == -1):
        i = i + 1
    elif (j == -1):
        j = j + 1
    else:
        i = i
        j = j
    M[i,j] = M[i,j] + 1
       

print("Minimum M:")      
print(np.min(M))
print("Maximum M:")    
print(np.max(M))
print("Mean of M:")    
print(np.mean(M))
print("Standard of deviation of M:")    
print(np.std(M))
       
L_size = 10
# Step size  is a 
M = np.zeros((10,10))
n_steps = 8000
location = np.zeros((2,))

location = random_location(L_size)
i = np.int(location[0])-1
j = np.int(location[1])-1
M[i,j] = M[i,j] + 1

for n in range(1,n_steps):
    val = random.randint(1, 4) 
    if val == 1: 
        i = i + 1
        j = j 
    elif val == 2: 
        i = i
        j = j + 1
    elif val == 3: 
        i = i
        j = j - 1
    else: 
        i = i - 1
        j = j 
    if (i == 10):
        i = i - 1
    elif (j == 10):
        j = j -1
    elif (i == -1):
        i = i + 1
    elif (j == -1):
        j = j+1
    else:
        i = i
        j = j
    M[i,j] = M[i,j] + 1
       

print("Minimum M:")      
print(np.min(M))
print("Maximum M:")    
print(np.max(M))
print("Mean of M:")    
print(np.mean(M))
print("Mean of M^2:")    
print(np.mean(M**2))
print("Standard of deviation of M:")        
print(np.sqrt(np.mean(M**2)-(np.mean(M))**2))
print("Standard of deviation of M:")    
print(np.std(M))
L_size = 10
n_list = [1000, 12000,23000,34000,45000,56000,67000,78000,89000,100000]
k = 0
pp = np.zeros((10,))
for n_s in n_list:
    
# Step size  is a 
    M = np.zeros((10,10))
    n_steps = n_s
    location = np.zeros((2,))
    location = random_location(L_size)
    i = np.int(location[0])-1
    j = np.int(location[1])-1
    M[i,j] = M[i,j] + 1

    for n in range(1,n_steps):
        val = random.randint(1, 4) 
        if val == 1: 
            i = i + 1
            j = j 
        elif val == 2: 
            i = i
            j = j + 1
        elif val == 3: 
            i = i
            j = j - 1
        else: 
            i = i - 1
            j = j 
        if (i == 10):
            i = i - 1
        elif (j == 10):
            j = j -1
        elif (i == -1):
            i = i + 1
        elif (j == -1):
            j = j+1
        else:
            i = i
            j = j
        M[i,j] = M[i,j] + 1
    M_m = np.mean(M)
    M_std = np.std(M)
    cov = M_std/M_m
    pp[k] = cov
    k += 1

f = plt.figure(figsize = (7.5,7.5))
plt.plot(n_list, pp, '.k')
plt.ylabel("$\sigma$/<M>")
plt.xlabel("Number of steps, $N_{steps}$")
plt.xlim(0,100000)
plt.ylim(0,1)
f.savefig('Confined_Lattice.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()