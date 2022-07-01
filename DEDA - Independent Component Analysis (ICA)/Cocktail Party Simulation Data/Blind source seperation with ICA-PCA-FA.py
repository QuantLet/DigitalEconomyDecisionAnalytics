#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 18:35:43 2022

@author: jane_hsieh
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile

#c
#import utilities as utl


# ======================================= 0. Input Data =========================================================
"""
Source of sound data: https://github.com/vishwajeet97/Cocktail-Party-Problem/blob/master/sounds
    - from there you can know the detail about how the source signals are mixed toghether, specifically in files:
        1. mixing_sound.py
        2. utilities.py (with 'mixSounds' function specified)
"""
# Read the source data as numpy array
rateY1, Y1 = wavfile.read("./Sound data/sourceX.wav") # = sourceX.wav
rateY2, Y2 = wavfile.read("./Sound data/sourceY.wav") # = sourceY.wav

# Read the mixing data as numpy array
rateX1, X1 = wavfile.read("./Sound data/mixedX.wav") # = sourceX.wav
rateX2, X2 = wavfile.read("./Sound data/mixedY.wav") # = sourceY.wav


length = X1.shape[0] / rateX1
print(f"length of X1 = {length}s") # 15.84
length = X2.shape[0] / rateX2
print(f"length of X2 = {length}s") # 15.84


# Visualization of source and mixing data  --------------------------------------------------------------
time = np.linspace(0., length, X1.shape[0])

fig, ax = plt.subplots(2, 2, sharex=True,  figsize=(15,15)) #sharey=True,
# True source signal Y
ax[0,0].set_title('Source $Y_1$')
ax[0,0].plot(time, Y1, label="Source Y1", color = 'k')
ax[0,1].set_title('Source $Y_2$')
ax[0,1].plot(time, Y2, label="Source Y2", color = 'k')
fig.text(0.5, 0.90, 'Source Signals', ha='center', color='k', fontsize=12)

# Mixed signals X
ax[1,0].set_title('Mixed $X_1$ (= $0.3*Y_1 + 0.7*Y_2$)')
ax[1,0].plot(time, X1, label="Mixed X1", color = 'b')
ax[1,1].set_title('Mixed $X_2$ (= $0.6*Y_1 + 0.4*Y_2$)')
ax[1,1].plot(time, X2, label="Mixed X2", color = 'b')

fig.text(0.5, 0.49, 'Mixed Signals', ha='center', color='b', fontsize=12)


fig.text(0.5, 0.04, 'Time [s]', ha='center')
fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')


plt.suptitle("A comparision of source and mixed signals", fontsize=14, fontweight ='bold')
plt.show()
plt.savefig("./Plots/Compare source and mixing data.png", transparent=True)







# ======================================= 1. Data Analysis: Blind Source Seperation =========================================================


## -------------------------------------- 1.1 ICA Analysis --------------------------------------------------------------
### Preprocessing Data --------------------------------------------------------------
# Centering the mixed signals

X1_m = X1 - np.mean(X1)
X2_m = X2 - np.mean(X2)
# Recaling of variance (i.e., standardization) of X1, X2 is not performed here since in later analysis,
# "whitening" would be operated instead

# Creating a matrix out of the signals
matrix = np.vstack([X1_m, X2_m]).T

### ICA --------------------------------------------------------------
from sklearn.decomposition import FastICA

transformer = FastICA(n_components=2, random_state=0, whiten='unit-variance')
Y_pred = transformer.fit_transform(matrix)
Y_pred.shape

"""
### Visualization --------------------------------------------------------------
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
# Plot seperated signals
ax[0].plot(time, Y_pred[:,0], label="IC Y1", color = 'navy')
ax[1].plot(time, Y_pred[:,1], label="IC Y2", color = 'navy')
plt.suptitle("The estimates (ICs) of the original source signals by ICA")
plt.show()
plt.savefig("ICA estimates of source data.png", transparency=True)
"""


'''
#Error4: generate wav file without sounds
scale=int(abs(matrix).max()/abs(Y_pred).max())
Y_pred_scale = (scale*Y_pred).astype(np.int16)

# Write the separated sound signals, 5000 is multiplied so that signal is audible
wavfile.write("./Output sound data/" + "ICA_Y1.wav", rateX1, Y_pred_scale[:,0])
wavfile.write("./Output sound data/" + "ICA_Y2.wav", rateX1, Y_pred_scale[:,1])
'''

### Output into audio file (wav) --------------------------------------------------------------
Max = np.max(np.abs(Y_pred))
Y_pred_f32 = (Y_pred/Max).astype(np.float32)

# Write the separated sound signals, 5000 is multiplied so that signal is audible
wavfile.write("./Output sound data/" + "ICA_Y1.wav", rateX1, Y_pred_f32[:,0])
wavfile.write("./Output sound data/" + "ICA_Y2.wav", rateX1, Y_pred_f32[:,1])



## -------------------------------------- 1.2 PCA Analysis --------------------------------------------------------------
### Preprocessing Data --------------------------------------------------------------
# Standardize the mixed signals (i.e., mean=0, std=1)
from sklearn.preprocessing import StandardScaler

matrix = np.vstack([X1, X2]).T

sc_X = StandardScaler()
matrix = sc_X.fit_transform(matrix)
'''#check
matrix.mean(axis=0) #>> Out[148]: array([ 3.24515947e-18, -1.56931146e-17])

matrix.std(axis=0)  #>> Out[149]: array([1., 1.])
'''

### PCA --------------------------------------------------------------
from sklearn.decomposition import PCA
pca = PCA(n_components=2) #n_components=None or integer
Y_pred_PCA = pca.fit_transform(matrix)
#PCs = pca.components_ #each row is a PC -> PC1 = pca.components_[0,:]

#Rescale Y_pred_PCA to the original scale as that of matrix (X)
Y_pred_PCA_rs = sc_X.inverse_transform(Y_pred_PCA)

"""
### Visualization --------------------------------------------------------------
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
# Plot seperated signals
ax[0].plot(time, Y_pred_PCA_rs[:,0], label="PC Y1", color = 'r')
ax[1].plot(time, Y_pred_PCA_rs[:,1], label="PC Y2", color = 'r')
plt.suptitle("The estimates (PCs) of the original source signals by PCA")
plt.show()
plt.savefig("PCA estimates of source data.png", transparency=True)
"""

### Output into audio file (wav) --------------------------------------------------------------
Max = np.max(np.abs(Y_pred_PCA_rs))
Y_pred_PCA_f32 = (Y_pred_PCA_rs/Max).astype(np.float32)

# Write the separated sound signals, 5000 is multiplied so that signal is audible
wavfile.write("./Output sound data/" + "PCA_Y1.wav", rateX1, Y_pred_PCA_f32[:,0])
wavfile.write("./Output sound data/" + "PCA_Y2.wav", rateX1, Y_pred_PCA_f32[:,1])


## -------------------------------------- 1.3. FA Analysis --------------------------------------------------------------
### Preprocessing Data --------------------------------------------------------------
matrix = np.vstack([X1, X2]).T

from sklearn.decomposition import FactorAnalysis
FA = FactorAnalysis(n_components=2, rotation='varimax', random_state=0)
Y_pred_FA = FA.fit_transform(matrix)
FA.get_params()
Y_pred_FA.shape

"""
### Visualization --------------------------------------------------------------
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
# Plot seperated signals
ax[0].plot(time, Y_pred_FA[:,0], label="Factor Y1", color = 'green')
ax[1].plot(time, Y_pred_FA[:,1], label="Factor Y2", color = 'green')
plt.suptitle("The estimates (PCs) of the original source signals by FA")
plt.show()
plt.savefig("FA estimates of source data.png", transparency=True)
"""

### Output into audio file (wav) --------------------------------------------------------------
Max = np.max(np.abs(Y_pred_FA))
Y_pred_FA_f32 = (Y_pred_FA/Max).astype(np.float32)

# Write the separated sound signals, 5000 is multiplied so that signal is audible
wavfile.write("./Output sound data/" + "FA_Y1.wav", rateX1, Y_pred_FA_f32[:,0])
wavfile.write("./Output sound data/" + "FA_Y2.wav", rateX1, Y_pred_FA_f32[:,1])



# ======================================= 2. Summary & Visualization of Results =========================================================

### Visualization of Results from PCA/FA/ICA --------------------------------------------------------------
fig, ax = plt.subplots(3, 2, sharex=True,  figsize=(15,10)) #sharey=True,
# Plot seperated signals: PCA
ax[0,0].plot(time, Y_pred_PCA_rs[:,0], label="PC Y1", color = 'r')
ax[0,1].plot(time, Y_pred_PCA_rs[:,1], label="PC Y2", color = 'r')
fig.text(0.5, 0.90, 'PCA results', ha='center', color='r', fontsize=12, fontweight='bold')

# Plot seperated signals: FA
ax[1,0].plot(time, Y_pred_FA[:,0], label="Factor Y1", color = 'green')
ax[1,1].plot(time, Y_pred_FA[:,1], label="Factor Y2", color = 'green')
fig.text(0.5, 0.62, 'FA results', ha='center', color='green', fontsize=12, fontweight='bold')

# Plot seperated signals: ICA
ax[2,0].plot(time, Y_pred[:,0], label="IC Y1", color = 'navy')
ax[2,1].plot(time, Y_pred[:,1], label="IC Y2", color = 'navy')
fig.text(0.5, 0.35, 'ICA results', ha='center', color='navy', fontsize=12, fontweight='bold')

fig.text(0.5, 0.04, 'Time [s]', ha='center')
fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')


plt.suptitle("A comparision of seperated signals by CFA, FA and ICA", fontsize=14, fontweight ='bold')
plt.show()
plt.savefig("./Plots/Compare estimates of source data from PCA-FA-ICA.png", transparent=True)



