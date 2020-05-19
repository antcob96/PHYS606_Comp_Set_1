# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:20:26 2020

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


def random_location(Length):
    location = random.random()*Length 
    return location
def F(x):
    force = -np.sin(np.pi*(x-50)/50)
    return force

def E(x):
    energy = -(50/np.pi)*np.cos(np.pi*(x-50)/50)
    return energy 

def delta_E(x,x_n): 
    delta_energy = E(x_n)-E(x)
    return delta_energy

def estimate_coef(x, y): 

  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum((x-m_x)*(y-m_y))
    SS_xx = np.sum((x-m_x)**2) 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1)

x_bin = np.linspace(1,100,100)
L = 100
x = np.zeros((100,))
i = np.int(random_location(L))-1
i_old = i
n_steps = 100000
beta = 1
for n in range(1,n_steps+1):
    # Randomly walk
    val = random.randint(1, 2) 
    if val == 1: 
        i = i + 1
    else: 
        i = i - 1
    # Figure out if its outside or not
    if i == -1: 
        i = i + 1
    elif i == 100: 
        i = i - 1
    else: 
        i = i
    # Compare energy of the new state with the old state
    d_E = delta_E(i_old+1,i+1)
    # Figure out its an acceptable state
    if  d_E <= 0:
        x[i] = x[i] + 1
        i_old = i
    else:
        if random.random() < np.exp(-beta*d_E):
            x[i] = x[i] + 1
            i_old = i
        else:
            x[i_old] = x[i_old] + 1
            i = i_old
           


f = plt.figure(figsize = (7.5,7.5))
plt.bar(x_bin,x)
plt.title('Histogram at Beta = 1')
plt.vlines(50,0, np.max(x),'k', linestyles='dashed')
plt.xlim(0,100)
plt.ylabel("Vists")
plt.xlabel("x")
#plt.ylim(0,100)
f.savefig('Histogram_b1_cs1_2a.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()
prob_b1 = x/n_steps

x = np.zeros((100,))
i = np.int(random_location(L))-1
i_old = i
n_steps = 100000
beta = 1/0.3
for n in range(1,n_steps+1):
    # Randomly walk
    val = random.randint(1, 2) 
    if val == 1: 
        i = i + 1
    else: 
        i = i - 1
    # Figure out if its outside or not
    if i == -1: 
        i = i + 1
    elif i == 100: 
        i = i - 1
    else: 
        i = i
    # Compare energy of the new state with the old state
    d_E = delta_E(i_old+1,i+1)
    # Figure out its an acceptable state
    if  d_E <= 0:
        x[i] = x[i] + 1
        i_old = i
    else:
        if random.random() < np.exp(-beta*d_E):
            x[i] = x[i] + 1
            i_old = i
        else:
            x[i_old] = x[i_old] + 1
            i = i_old
           

#his, bi = np.histogram(x)
#plt.hist(his)
#plt.show()
f = plt.figure(figsize = (7.5,7.5))
plt.bar(x_bin,x)
plt.title('Histogram at Beta = 10/3')
plt.vlines(50,0, np.max(x),'k', linestyles='dashed')
plt.xlim(0,100)
plt.ylabel("Visits")
plt.xlabel("x")
#plt.ylim(0,100)
f.savefig('Histogram_b2_cs1_2a.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()
prob_b2 = x/n_steps

x = np.zeros((100,))
i = np.int(random_location(L))-1
i_old = i
n_steps = 100000
beta = 1/0.1
for n in range(1,n_steps+1):
    # Randomly walk
    val = random.randint(1, 2) 
    if val == 1: 
        i = i + 1
    else: 
        i = i - 1
    # Figure out if its outside or not
    if i == -1: 
        i = i + 1
    elif i == 100: 
        i = i - 1
    else: 
        i = i
    # Compare energy of the new state with the old state
    d_E = delta_E(i_old+1,i+1)
    # Figure out its an acceptable state
    if  d_E <= 0:
        x[i] = x[i] + 1
        i_old = i
    else:
        if random.random() < np.exp(-beta*d_E):
            x[i] = x[i] + 1
            i_old = i
        else:
            x[i_old] = x[i_old] + 1
            i = i_old
           

#his, bi = np.histogram(x)
#plt.hist(his)
#plt.show()
f = plt.figure(figsize = (7.5,7.5))
plt.vlines(50,0, np.max(x),'k', linestyles='dashed')
plt.bar(x_bin,x)
plt.title('Histogram at Beta = 10')
plt.xlim(0,100)
plt.ylabel("Vists")
plt.xlabel("x")
#plt.ylim(0,100)
f.savefig('Histogram_b3_cs1_2a.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()


prob_b3 = x/n_steps


F_b1 = np.zeros((100,))
F_b2 = np.zeros((100,))
F_b3 = np.zeros((100,))
for i in range(0,99):
    x_i = i + 1
    F_b1[i] = prob_b1[i]*np.abs(F(x_i))
    F_b2[i] = prob_b2[i]*np.abs(F(x_i))
    F_b3[i] = prob_b3[i]*np.abs(F(x_i))

avg_F_b1 = 0
avg_F_b2 = 0
avg_F_b3 = 0
for i in range(0,99):
    avg_F_b1 = avg_F_b1 + F_b1[i]
    avg_F_b2 = avg_F_b2 + F_b2[i]
    avg_F_b3 = avg_F_b3 + F_b3[i]
#avg_F_b1 = np.mean(F_b1,dtype = np.float64)


print("Average of |F| with beta^-1 = 1:")
print(avg_F_b1)

#avg_F_b2 = np.mean(F_b2,dtype = np.float64)
print("Average of |F| with beta^-1 = 0.3:")
print(avg_F_b2)

#avg_F_b3 = np.mean(F_b3,dtype = np.float64)
print("Average of |F| with beta^-1 = 0.1:")
print(avg_F_b3)
N_c = 50
cor = np.zeros((500,200))
cor_avg = np.zeros((500,))
for r in range(0,199):
    n_steps = 500 
    x = np.zeros((100,))
    i = np.int(random_location(L))-1
    i_old = i
    x_0 = i
    n_steps = 500
    beta = 1/0.3
    for n in range(0,n_steps):
        cor[n,r] = cor[n,r] + (x_0-50)*(i-50)
        # Randomly walk
        val = random.randint(1, 2) 
        if val == 1: 
            i = i + 1
        else: 
            i = i - 1
        # Figure out if its outside or not
        if i == -1: 
            i = i + 1
        elif i == 100: 
            i = i - 1
        else: 
            i = i
        # Compare energy of the new state with the old state
        d_E = delta_E(i_old+1,i+1)
        # Figure out its an acceptable state
        if  d_E <= 0:
            x[i] = x[i] + 1
            i_old = i
        else:
            if random.random() < np.exp(-beta*d_E):
                x[i] = x[i] + 1
                i_old = i
            else:
                x[i_old] = x[i_old] + 1
                i = i_old
                
for i in range(0,500):
    cor_avg[i] = np.mean(cor[i,:])        
f = plt.figure(figsize = (7.5,7.5))
time = np.linspace(0,500, 500)
plt.plot(time, cor_avg)
plt.xlim(0,500)
plt.ylim(0,np.max(cor_avg))
plt.xlabel("Time, t")
plt.ylabel("Correlation Function, C(t)")
plt.title('Correlation function versus time')
f.savefig('Correlation_cs1_2c.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()



cor = np.zeros((500,200))
cor_avg = np.zeros((500,))
for r in range(0,199):
    n_steps = 500 
    x = np.zeros((100,))
    i = random.randint(40, 60) 
    i_old = i
    x_0 = i
    n_steps = 500
    beta = 1/0.3
    for n in range(0,n_steps):
        cor[n,r] = cor[n,r] + (x_0-50)*(i-50)
        # Randomly walk
        val = random.randint(1, 2) 
        if val == 1: 
            i = i + 1
        else: 
            i = i - 1
        # Figure out if its outside or not
        if i == -1: 
            i = i + 1
        elif i == 100: 
            i = i - 1
        else: 
            i = i
        # Compare energy of the new state with the old state
        d_E = delta_E(i_old+1,i+1)
        # Figure out its an acceptable state
        if  d_E <= 0:
            x[i] = x[i] + 1
            i_old = i
        else:
            if random.random() < np.exp(-beta*d_E):
                x[i] = x[i] + 1
                i_old = i
            else:
                x[i_old] = x[i_old] + 1
                i = i_old
N_c = 40
#cor = cor/200
cor_test = np.zeros((N_c,))
time = np.linspace(0,N_c, N_c)
time_plot = np.linspace(0,500,500)
for i in range(0,N_c):
    cor_test[i] = np.mean(cor[i,:])
for i in range(0,500):
    cor_avg[i] = np.mean(cor[i,:])         
    
model = LinearRegression(fit_intercept=True)

model.fit(time[:, np.newaxis], np.log(cor_test))
yfit = model.predict(time[:,np.newaxis])
tauinv = model.coef_[0]
tau = 1/tauinv
print("Decay Time:")
print(tau)
cor_fit = cor_test[0]*np.exp(tauinv*time_plot)
f = plt.figure(figsize = (7.5,7.5))
plt.plot(time_plot,cor_fit,'-r', label = 'fit')
plt.plot(time_plot,cor_avg, '-k', label = 'Data')
plt.xlim(0,500)
#plt.ylim()
plt.xlabel("Time, t")
plt.ylabel("Correlation Function, C(t)")
plt.title('Best fit of expoential decay')
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
f.savefig('Cor_decay_cs1_2d.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()