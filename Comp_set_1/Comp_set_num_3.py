# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:31:27 2020

@author: tonyc
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:19:57 2020

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
import seaborn as sns
from mpl_toolkits import mplot3d

import pandas as pd
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.linear_model import LinearRegression
import time



def energy(state,mu):
# Define the energy of
    E = - mu*state
    return E

def delta_E(x,x_n,mu): 
    delta_energy = energy(x_n,mu)-energy(x,mu)
    return delta_energy


#b_list = np.array([0.0001,0.001,0.01,0.1,1,10])
L = 64
mu = 1
var_avg_steps = 1000
var_steps = 61
e = (var_steps - 1)/5
n_trans = 4000
n_comp = 1000
var_array = np.zeros((var_steps,))
b_list = np.zeros((var_steps,))
avgvar = np.zeros((var_avg_steps,var_steps))
Super_var = np.zeros((var_steps,))
start_time = time.time()
for d_v in range(1,var_avg_steps+1):
    ll = 0  
    
    for b_p in range(1,var_steps+1): 
        beta = 10**(-4+(b_p-1)/e)
        b_list[b_p-1] = beta
        N_total = np.zeros((n_comp,))
        state_old = np.zeros((L,))
        state_trial = np.zeros((L,))
        for u in range(0,L-1):
            state_old[u] = random.randint(0,1)
            state_trial[u] = state_old[u]
        #print("out",state_old)
        for n in range(1,n_trans+1):
            i = random.randint(0,L-1)
            state_o = state_old[i]
            if state_o == 0:
                state_trial[i] = 1
                
                #print('new',state_trial[i],'old',state_old[i],'actual', state_o)
                state_old[i] = 1
                #print(state_old[i],state_trial[i])
            elif state_o == 1:
                state_trial[i] = 0
                #print('new',state_trial[i],'old',state_old[i],'actual', state_o)
                #print(state_old[i],state_trial[i])
                sum_old = np.sum(state_old)
                sum_trial = np.sum(state_trial)
                dE = delta_E(sum_old, sum_trial, mu)
                if np.random.random() < np.exp(-beta*dE):
                    state_old[i] = 0
                    #print("Flipped")
                else:
                    #print("Didn't")
                    state_trial[i] = 1
            else:
                print("Something")
                break
        for u in range(1,n_comp+1):
            i = random.randint(0,L-1)
            
            state_0 = state_old[i]
           # print("Before",state_0, state_old[i], state_trial[i])
            if state_0 == 0:
                state_trial[i] = 1
                #print('new',state_trial[i],'old',state_old[i],'actual', state_0)
                state_old[i] = 1
            else:
                state_trial[i] = 0
                #print('new',state_trial[i],'old',state_old[i],'actual', state_0)
                sum_old = np.sum(state_old)
                sum_trial = np.sum(state_trial)
                dE = delta_E(sum_old, sum_trial, mu)
                if random.random() < np.exp(-beta*dE):
                    state_old[i] = 0
                else:
                    state_trial[i] = 1
            N_total[u-1] = np.sum(state_trial) 
#        sum_var = 0
#        for uuu in range(1,n_comp+1):
#            sum_var = sum_var + N_total[uuu-1]
#        mean_var = sum_var/n_comp
#       var_sum = 0
#        for uu in range(1,n_comp+1):
#            var_sum = var_sum + (N_total[uu-1])**2
#        mean_var_sq = var_sum/n_comp
        var_array[ll] = np.var(N_total) #mean_var_sq - mean_var**2
        avgvar[d_v-1,ll] = var_array[ll]
        ll += 1

t_time = (time.time() - start_time)
t_pred =  2.4*var_avg_steps
print("Actual time:",t_time/60,'mins')
print("Predicted Time:",t_pred/60, "mins")
print("Difference:",np.abs((t_pred - t_time)/t_pred)*100, "%")
for ui in range(1,var_steps+1):
    Super_var[ui-1] = np.mean(avgvar[:,ui-1])
f = plt.figure(figsize = (7.5,7.5))
plt.semilogx(1/(b_list),Super_var,'-k')
plt.xlim(1e-1,1/1e-4)
plt.ylabel("Variance of N, Var(N)")
plt.xlabel("Temperature, T")
plt.ylim(0,20)
f.savefig('Variance_cs1_3.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()