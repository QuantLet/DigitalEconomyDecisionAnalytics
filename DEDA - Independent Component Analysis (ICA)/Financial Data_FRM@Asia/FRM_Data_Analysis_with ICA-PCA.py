#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/06/29

@author: jane_hsieh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

os.getcwd()
os.chdir("/Users/jane_hsieh/Library/CloudStorage/OneDrive-國立陽明交通大學/Data Science Analysis Templates/Machine Learning/Part 9 - Dimensionality Reduction/SectionExtra 3 - Independent Component Analysis (ICA)/Financial Data_FRM@Asia")  


data_dir = './Data'
output_dir = './Output'

# ====================================  0. Input data: FRM prices ====================================================
#df0 = pd.read_csv(data_dir+'/FRM_SHSZ300HSITWSE_Stock_Prices_update_20201030.csv', parse_dates=['Date'], index_col = 'Date')
df0 = pd.read_csv(data_dir+'/FRM_CHHKTW_Time_Series_Returns_20201030.csv', parse_dates=['Date'], index_col = 'Date')


col_chosen = [28, 159, 172, 182, 153, 108, 41, 154, 107, 173, 27, 122, 64, 163, 105] #[28, 159, 172, 182, 153] #
col_chosen = [i-1 for i in col_chosen]

#df0 = df0.iloc[:,col_chosen]
#df0.columns

df = df0.iloc[:,col_chosen]
#df = df0

df.columns = [i.replace('.EQUITY', '') for i in df.columns]
df.columns


## 0.1 Missing data imputation ---------------------------------------------------------------
print("Numnber of missing data: \n", df.isnull().sum())
'''
#Only sporatic missing points, hence we simply perform linear interpolation method to those missing values

df.interpolate(method='linear', inplace=True)
'''


## 0.2 Visualization: Multidimensional Time Series Data Plot (FRM) ---------------------------------------------------------------
start = '2019-01'
end =  '2020-10'

df[start: end].plot(figsize=(15,6))
#plt.legend(fancybox=True, framealpha=0.0, loc = 'upper right', prop={'size': 8})
plt.legend(bbox_to_anchor=(1.02, 1),loc='upper left', borderaxespad=0., 
           fancybox=True, framealpha=0.0, prop={'size': 8})
plt.title(f'Time Series Returns (FRM) from {start} to {end}')

plt.show()
plt.savefig(data_dir+f'/Multidimensional Time Series Returns (FRM) from {start} to {end} ({len(col_chosen)}D).png', transparent = True)
plt.close()








# ======================================= 1. Data Analysis: Portfolio construction (via PCA vs ICA) =========================================================

# Number of components
k = 5
## -------------------------------------- 1.1 ICA Analysis --------------------------------------------------------------

### ICA --------------------------------------------------------------
from sklearn.decomposition import FastICA

transformer = FastICA(n_components=k, random_state=0, whiten='unit-variance')
Y_pred = transformer.fit_transform(df)
Y_pred.shape

Y_pred = pd.DataFrame(Y_pred)
Y_pred.columns= ['IC'+str(i+1) for i in range(Y_pred.shape[1])]
Y_pred.set_index(df.index, inplace=True)

'''
### Visualization: TS for ICs ---------------------------------------------------------------
Y_pred.loc[start: end].plot(figsize=(15,6))
plt.legend(fancybox=True, framealpha=0.0, loc = 'upper right', prop={'size': 8})
plt.title('Time Series Returns of ICs')
plt.show()
plt.savefig(output_dir+f'/Time Series Returns of {k} ICs ({int(df.shape[1])}D).png', transparent = True)
plt.close()
'''


### Visualization: W (weight matrix to transform raw data X into ICs Y) ---------------------------------------------------------------
# Investigate W
W_IC  = pd.DataFrame(transformer.components_, columns=df.columns)
W_IC.set_index(Y_pred.columns, inplace=True)
#W_IC.to_csv(output_dir+f'/W matrix of {k} ICs ({int(df.shape[1])}D).csv')

'''
import seaborn as sns
# Customize the heatmap oW weight matrix
sns.set(rc={'figure.figsize':(15,4)})
sns.heatmap(W_IC,
            annot=True,
            center = 0,
            linewidths=0.4,
            annot_kws={"size": 10})
plt.title(f'Portfolio (weights) construction with {k} ICA')
plt.xticks(rotation=45, fontsize=7)
plt.yticks(rotation=0) 
plt.show()
plt.savefig(output_dir+f'/W matrix of {k} ICs ({int(df.shape[1])}D).png', transparent = True)
plt.close()
'''

## -------------------------------------- 4.2 PCA Analysis --------------------------------------------------------------
### Preprocessing Data --------------------------------------------------------------
# Standardize the mixed signals (i.e., mean=0, std=1)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
matrix = sc_X.fit_transform(df)


'''#check
matrix.mean(axis=0) 
matrix.std(axis=0) 
'''

### PCA --------------------------------------------------------------
from sklearn.decomposition import PCA
pca = PCA(n_components=k) #n_components=None or integer
Y_pred_PCA = pca.fit_transform(matrix)

Y_pred_PCA = pd.DataFrame(Y_pred_PCA)
Y_pred_PCA.columns= ['PC'+str(i+1) for i in range(k)]
Y_pred_PCA.set_index(df.index, inplace=True)

#Change direction of PC1 (only for better visualization, won't change results)
Y_pred_PCA.loc[:, 'PC1'] =  -Y_pred_PCA.loc[:, 'PC1']

#Rescale Y_pred_PCA to the original scale as that of matrix (X)
#X_mean = sc_X.mean_[:k]
#X_std = np.sqrt(sc_X.var_[:k])
#Y_pred_PCA_rs = (Y_pred_PCA * X_std) + X_mean #sc_X.inverse_transform(Y_pred_PCA)


'''
### Visualization: PCs ---------------------------------------------------------------

Y_pred_PCA.loc[start: end].plot(figsize=(15,6))
plt.legend(fancybox=True, framealpha=0.0, loc = 'upper right', prop={'size': 8})
plt.title('Time Series Returns of PCs')
plt.show()
plt.savefig(output_dir+f'/Time Series Returns of {k} PCs ({int(df.shape[1])}D).png', transparent = True)
plt.close()
'''

### Visualization: W (weight matrix to transform raw data X into ICs Y) ---------------------------------------------------------------
# Investigate W
W_PC  = pd.DataFrame(pca.components_, columns=df.columns) ##each row is a PC -> PC1 = pca.components_[0,:]
W_PC.set_index(Y_pred_PCA.columns, inplace=True)
#W_PC.to_csv(output_dir+f'/W matrix of {k} PCs ({int(df.shape[1])}D).csv')

#Similary, change sign of weights due fo change direction of PC1 (only for better visualization, won't change results)
W_PC.loc['PC1', :] =  -W_PC.loc['PC1', :]


'''
import seaborn as sns

# Customize the heatmap oW weight matrix
#sns.color_palette("vlag", as_cmap=True)
sns.heatmap(W_PC,
            annot=True,
            center = 0,
            linewidths=0.4,
            annot_kws={"size": 10})
plt.title(f'Portfolio (weights) construction with {k} PCA')
plt.xticks(rotation=30, fontsize=7)
plt.yticks(rotation=0) 
plt.show()
plt.savefig(output_dir+f'/W matrix of {k} PCs ({int(df.shape[1])}D).png', transparent = True)
plt.close()

'''



# ======================================= 5. Summary & Visualization of Results =========================================================

### Visualization of Results from PCA/FA/ICA --------------------------------------------------------------
fig, ax = plt.subplots(2, 1, sharex=True,  figsize=(30,12)) #sharey=True,
# Plot seperated signals: PCA
ax[0].plot(df.index, Y_pred_PCA.loc[start: end], label=Y_pred_PCA.columns)
ax[0].legend(fancybox=True, framealpha=0.0, loc = 'upper right', prop={'size': 8})
fig.text(0.5, 0.90, f'Returns of first {k} PCs', ha='center', fontsize=12)

# Plot seperated signals: ICA
ax[1].plot(df.index, Y_pred.loc[start: end], label=Y_pred.columns)
ax[1].legend(fancybox=True, framealpha=0.0, loc = 'upper right', prop={'size': 8})
fig.text(0.5, 0.48, f'Returns of first {k} ICs', ha='center', fontsize=12)


fig.text(0.5, 0.04, 'Date', ha='center', fontsize=14, fontweight ='bold')
fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical', fontsize=14, fontweight ='bold')


plt.suptitle("A comparision of seperated stock returns by PCA and ICA", fontsize=14, fontweight ='bold')
plt.savefig(output_dir+"/Compare estimates of returns from PCA-ICA.png", transparent=True)
plt.show()
plt.close()




### Heatmaps of W (weight) matrix from PCA/FA/ICA --------------------------------------------------------------
import seaborn as sns

fig, ax = plt.subplots(2, 1, sharex=True,  figsize=(30,12)) #sharey=True,
# Plot seperated signals: PCA
sns.heatmap(W_PC,
            annot=True,
            center = 0,
            linewidths=0.4,
            annot_kws={"size": 10}, ax = ax[0])
ax[0].legend(fancybox=True, framealpha=0.0, loc = 'upper right', prop={'size': 8})
fig.text(0.5, 0.90, f'Portfolio with first {k} PCs', ha='center', fontsize=12)

# Plot seperated signals: ICA
sns.heatmap(W_IC,
            annot=True,
            center = 0,
            linewidths=0.4,
            annot_kws={"size": 10}, ax = ax[1])
ax[1].legend(fancybox=True, framealpha=0.0, loc = 'upper right', prop={'size': 8})
fig.text(0.5, 0.48, f'Portfolio with first {k} ICs', ha='center', fontsize=12)


fig.text(0.5, 0.04, 'Equities', ha='center', fontsize=14, fontweight ='bold')
plt.xticks(rotation=20, fontsize=7)
plt.suptitle("Portfolio (weights) construction with PCA or ICA", fontsize=14, fontweight ='bold')
plt.savefig(output_dir+"/Compare estimates of W matrix from PCA-ICA.png", transparent=True)
plt.show()
plt.close()



