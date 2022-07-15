#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:10:30 2022

@author: jane_hsieh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir("/Users/jane_hsieh/Library/CloudStorage/OneDrive-國立陽明交通大學/Data Science Analysis Templates/Machine Learning/Part 9 - Dimensionality Reduction/SectionExtra 3 - Independent Component Analysis (ICA)/Negentropy approximation")  
#os.getcwd()

np.random.seed(2022)

# ====================================  0. Generate data  ====================================================


# Define function to generate y based on parameter p
def Generate_y(p, N):
    '''
    y ~ p*N(0,1)+ (1-p)*N(1,4) 
    
    input:
        p: parameter to generate y, p \in [0,1]
        N: sample size
    output:
        y: i.i.d. sample after standardization (i.e., E{y} = 0, E{y^2}=Var(y)=1)
    '''
    u = np.random.uniform(low=0.0, high=1.0, size=N)
    z1 = np.random.normal(loc=0.0, scale=1.0, size=N)
    z2 = np.random.normal(loc=1.0, scale=4.0, size=N) #np.random.normal(loc=10.0, scale=5.0, size=N)
    y = (u<=p)*z1 + (u>p)*z2
    y = (y-np.mean(y))/np.std(y)
    return y



N = 100000
y_gauss = Generate_y(1, N)
print(f'Mean and std of y_gauss is {np.mean(y_gauss):0.3f} and {np.std(y_gauss):0.3f}')

p = 0.2
y_p = Generate_y(p, N)
print(f'Mean and std of y is {np.mean(y_gauss):0.3f} and {np.std(y_gauss):0.3f}')


# ====================================  1. Approximate (Neg)entropy: sample-based, method a and b ====================================================
from scipy import stats
import math

# Sample-based ----------------------------------------------------------------------------------------
gauss_entropy_s = stats.differential_entropy(y_gauss)
print(f"sample-based differential entropy for y_gauss: {gauss_entropy_s:0.4f}")

entropy_s = stats.differential_entropy(y_p)
print(f"sample-based differential entropy for y: {entropy_s:0.4f}")

negentropy_s = gauss_entropy_s - entropy_s
print(f"sample-based differential negentropy for y: {negentropy_s:0.4f}")

del y_p

# method a ----------------------------------------------------------------------------------------

## Define contrast functions ￼G_i's
def G1(y):
    G_y = y * np.exp(-y**2/2)
    return G_y
    
def G2_a(y):
    G_y = np.exp(-y**2/2)
    return G_y

def G2_b(y):
    G_y = np.abs(y)
    return G_y
    

## Define function to approximate negentropy of y using method a -- J_a
def J_a(y):
    k1 = 36/(8*np.sqrt(3)-9)
    k2_a = 1/(2-6/math.pi)
    
    G1_y = G1(y)
    G2_a_y = G2_a(y)
    
    J = k1 * np.mean(G1_y)**2 + k2_a * (np.mean(G2_a_y) - np.sqrt(0.5))**2
    
    return J
    

#J_a(y_p)


# method b ----------------------------------------------------------------------------------------
## Define function to approximate negentropy of y using method b -- J_b
def J_b(y):
    k1 = 36/(8*np.sqrt(3)-9)
    k2_b = 24/(16*np.sqrt(3)-27/math.pi)
    
    G1_y = G1(y)
    G2_b_y = G2_b(y)
    
    J = k1 * np.mean(G1_y)**2 + k2_b * (np.mean(G2_b_y) - np.sqrt(2/math.pi))**2
    
    return J

#J_b(y_p)



# ====================================  3. Visualization of (Estimated) Entropies for p in [0, 1] ====================================================
np.random.seed(2022)
N = 50000
y_gauss = Generate_y(1, N)
gauss_entropy_s = stats.differential_entropy(y_gauss)

## Estimate Negentropies
ps = np.linspace(0,1,201);ps #101

Estimation = []

for p in ps:
    #print(p)
    
    y_p = Generate_y(p, N)
    
    
    # true negentropy (sample-based)
    entropy_s = stats.differential_entropy(y_p)    
    negentropy_s = gauss_entropy_s - entropy_s    
    
    # method a
    negentropy_a = J_a(y_p)
    
    # method b
    negentropy_b = J_b(y_p)
    
    #Append data
    Estimation.append([p, negentropy_s, negentropy_a, negentropy_b])
    

Estimation = pd.DataFrame(Estimation, columns = ['p', 'True', 'Method a', 'Method b'])  
Estimation.set_index('p', inplace=True)


## Find best p that maximize Negentropy
ps_best = []
for col in Estimation.columns:
    i = np.argmax(Estimation[col])
    p_best = ps[i]
    print(f'Best p that maximizes Negentropy from "{col}" value is {p_best:0.3f}')
    
    ps_best.append(p_best)

ps_best = pd.Series(ps_best, index = Estimation.columns)
    
p_s_best = ps[i]; print(f'Best p that maximizes true Negentropy value is {p_s_best:0.3f}')
i = np.argmax(Estimation['True'])
p_s_best = ps[i]; print(f'Best p that maximizes Negentropy value based on "True" is {p_s_best:0.3f}')

 

## Visualization
colors = ['k', 'r', 'g']

plt.figure(figsize=(12,6))
for i in range(3):
    plt.plot(Estimation.index, Estimation.iloc[:,i], c = colors[i], alpha = 0.5, label = Estimation.columns[i])
    p_best = ps_best[Estimation.columns[i]]
    plt.axvline(p_best,  linestyle='--', color= colors[i], alpha = 0.5) #'k',
    plt.annotate(r"$p_{max}\approx$"+f"{p_best:0.3f}", xy = (p_best, 0), xytext = (p_best, np.max(Estimation.iloc[:,i])), ha='center', color= colors[i])#
plt.legend(loc='upper left')
plt.xlabel('p')
plt.ylabel('Negentropy estimate')
plt.title(f'Negentropy Comparison (N={N})')
plt.savefig(f'Negentropy Comparison (N={N}).png', transparent = True)
plt.show()


