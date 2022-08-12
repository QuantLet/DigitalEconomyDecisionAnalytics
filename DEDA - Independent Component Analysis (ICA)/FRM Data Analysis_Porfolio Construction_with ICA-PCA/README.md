[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **FRM Data Analysis_Porfolio Construction_with ICA-PCA** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: 'FRM Data Analysis_Porfolio Construction_with ICA-PCA'

Published in: 'DEDA class 2022'

Description: 'Analyze financial data (daily returns of 15 stocks in Asia market) - show the difference between results of PCA and ICA'

Submitted:  '09 Aug 2022'

Keywords: 
- 'Independent Component Analysis (ICA)'
- 'Principal Component Analysis (PCA)'
- 'Profile Construction'
- 'Financial data (Stock market data)'
- 'Daily Returns'

Input: 'the module–"FRM Data Analysis_Porfolio Construction_with ICA-PCA.py" is used to analyzie financial data for portfolio construction, and show the different results of ICA and PCA.'

Datafile:
- 'FRM_CHHKTW_Time_Series_Returns_20201030.csv'
- 'FRM_SHSZ300HSITWSE_Stock_Prices_update_20201030.csv'


Output: 
- 'Multidimensional Daily Prices (FRM) from 2019-01-02 to 2020-10-30 (15D).png'
- 'Multidimensional Daily Returns (FRM) from 2019-01-02 to 2020-10-30 (15D).png'
- 'Compare estimates of (rescaled) W matrix from PCA-ICA.png'
- 'Compare estimates of portfolio returns from PCA-ICA.png'
- 'Compare estimates of portfolio returns from PCA-ICA 2.png'
- 'Compare cumulative returns from PCA-ICA.png'
- 'Compare cumulative returns from PCA-ICA 2.png'
- 'Test- Compare estimates of portfolio returns from PCA-ICA.png'
- 'Test- Compare cumulative returns from PCA-ICA.png'


Author: 
- 'Jane Hsieh (Hsing-Chuan Hsieh)'
- 'Wolfgang Karl Härdle'

```

![Picture1](Compare%20cumulative%20returns%20from%20PCA-ICA%202.png)

![Picture2](Compare%20cumulative%20returns%20from%20PCA-ICA.png)

![Picture3](Compare%20estimates%20of%20(rescaled)%20W%20matrix%20from%20PCA-ICA.png)

![Picture4](Compare%20estimates%20of%20portfolio%20returns%20from%20PCA-ICA%202.png)

![Picture5](Compare%20estimates%20of%20portfolio%20returns%20from%20PCA-ICA.png)

![Picture6](Multidimensional%20Daily%20Prices%20(FRM)%20from%202019-01-02%20to%202020-10-30%20(15D).png)

![Picture7](Multidimensional%20Daily%20Returns%20(FRM)%20from%202019-01-02%20to%202020-10-30%20(15D).png)

![Picture8](Test-%20Compare%20cumulative%20returns%20from%20PCA-ICA.png)

![Picture9](Test-%20Compare%20estimates%20of%20portfolio%20returns%20from%20PCA-ICA.png)

### PYTHON Code
```python

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/06/29

@author: jane_hsieh

@Resources:
    1. eigen-portfolio-construction with PCA: 
        https://github.com/titmy/eigen-portfolio-construction

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
import os

os.getcwd()
os.chdir("/Users/jane_hsieh/Library/CloudStorage/OneDrive-國立陽明交通大學/Data Science Analysis Templates/Machine Learning/Part 9 - Dimensionality Reduction/SectionExtra 3 - Independent Component Analysis (ICA)/FRM Data Analysis_Porfolio Construction_with ICA-PCA")  


data_dir = './Data'
output_dir = './Output'

# ====================================  0. Input data: FRM prices / returns ====================================================
# returns
df_return = pd.read_csv(data_dir+'/FRM_CHHKTW_Time_Series_Returns_20201030.csv', parse_dates=['Date'], index_col = 'Date')



col_chosen = [28, 159, 172, 182, 153, 108, 41, 154, 107, 173, 27, 122, 64, 163, 105] #[28, 159, 172, 182, 153] #
col_chosen = [i-1 for i in col_chosen]


stocks = df_return.columns[col_chosen];print(stocks)

#df_return = df_return.iloc[:,col_chosen]
#df_return.columns



df = df_return.iloc[:,col_chosen]
del df_return


df.columns = [i.replace('.EQUITY', '') for i in df.columns]
df.columns

# prices
df_price = pd.read_csv(data_dir+'/FRM_SHSZ300HSITWSE_Stock_Prices_update_20201030.csv', parse_dates=['Date'], index_col = 'Date')

stocks2 = df_price.columns[col_chosen];print(stocks2) #check if stocks2== stocks

df_price = df_price.iloc[:,col_chosen]
df_price.columns = df.columns


## 0.1 Missing data imputation ---------------------------------------------------------------
print("Numnber of missing data for returns:: \n", df.isnull().sum())
print("Numnber of missing data for prices: \n", df_price.isnull().sum())
'''
#Only sporatic missing points, hence we simply perform linear interpolation method to those missing values

df_return.fillna(0) #since supposed stock price has no change for missing value; i.e., df_price.fillna(method='ffill', inplace=True)
'''
df_price.fillna(method='ffill', inplace=True)




## 0.2 Visualization: Multidimensional Time Series Data Plot (FRM) ---------------------------------------------------------------
start = '2019-01-02'
end =  '2020-10-30'

df[start: end].plot(figsize=(15,6))
#plt.legend(fancybox=True, framealpha=0.0, loc = 'upper right', prop={'size': 8})
plt.legend(bbox_to_anchor=(1.02, 1),loc='upper left', borderaxespad=0., 
           fancybox=True, framealpha=0.0, prop={'size': 8})
plt.title(f'Daily returns of each stock (FRM@Asia) from {start} to {end}')

plt.show()
plt.savefig(data_dir+f'/Multidimensional Daily Returns (FRM) from {start} to {end} ({len(col_chosen)}D).png', transparent = True)
plt.close()


df_price[start: end].plot(figsize=(15,6))
#plt.legend(fancybox=True, framealpha=0.0, loc = 'upper right', prop={'size': 8})
plt.legend(bbox_to_anchor=(1.02, 1),loc='upper left', borderaxespad=0., 
           fancybox=True, framealpha=0.0, prop={'size': 8})
plt.title(f'Daily prices of each stock (FRM@Asia) from {start} to {end}')

plt.show()
plt.savefig(data_dir+f'/Multidimensional Daily Prices (FRM) from {start} to {end} ({len(col_chosen)}D).png', transparent = True)
plt.close()



# ======================================= 1. Data Analysis: Portfolio construction (via PCA vs ICA) =========================================================

## -------------------------------------- Functions --------------------------------------------------------------
def Rescale_W(W):
    '''
    input: W_IC - dataframe; weigh matrix with dim=(n_components, n_variables)
    output: W_IC _rs - dataframe; scaling  each w (row) to sum to 1 
    '''
    W_rs = list()
    for w in W.values:
        #print(w)
        # scaling  values to sum to 1 
        normalized_values = w  / w.sum() 
        W_rs.append(normalized_values)
    W_rs = pd.DataFrame(W_rs, columns=W.columns, index=W.index)
    return W_rs


## Truncate weights
def Trunc(x, threshold=1):
    if x>=threshold:
        xt = threshold
    elif x<=-threshold:
        xt = -threshold
    else:
        xt=x
    return xt

def sharpe_ratio(ts_returns, periods_per_year=252):
    """
    sharpe_ratio - Calculates annualized return, annualized vol, and annualized sharpe ratio, 
                    where sharpe ratio is defined as annualized return divided by annualized volatility 
                    
    Arguments:
    ts_returns - pd.Series of returns of a single eigen portfolio
    
    Return:
    a tuple of three doubles: annualized return, volatility, and sharpe ratio
    """
    
    annualized_return = 0.
    annualized_vol = 0.
    annualized_sharpe = 0.
    
    n_years = ts_returns.shape[0] / periods_per_year
    annualized_return = np.power(np.prod(1+ts_returns),(1/n_years))-1
    
    annualized_vol = ts_returns.std() * np.sqrt(periods_per_year)

    annualized_sharpe = annualized_return / annualized_vol
    
    mean = ts_returns.mean()
    var = ts_returns.var()
    k =  kurtosis(ts_returns)
    
    return annualized_return, annualized_sharpe, annualized_vol, mean, var, k #Ann_return, Ann_Sharpe, annualized_vol, mean, volatility(var), kurtosis




## -------------------------------------- 1.0 Preprocessing --------------------------------------------------------------
## Split df in training and testing set
#import datetime

train_ratio = 0.85
train_end = int(len(df)*train_ratio)#datetime.datetime(2012, 3, 26) 



# for daily returns
df_train = df[:train_end].copy(); print('Dim. of Train dataset:', df_train.shape)
df_test = df[train_end:].copy(); print('Dim. of Test dataset:', df_test.shape)

print('Train dataset:', df_train.shape)
print('Test dataset:', df_test.shape)



# Standardize the mixed signals (i.e., mean=0, std=1)
from sklearn.preprocessing import StandardScaler

sc_df = StandardScaler()
df_train_rs = sc_df.fit_transform(df_train)
df_train_rs = pd.DataFrame(df_train_rs, columns=df.columns, index = df.index[:train_end])

print('Check the means of training data:\n', df_train_rs .mean(axis=0) )
print('Check the SDs of training data:\n',df_train_rs.std(axis=0) )


# for daily prices
train_end_date = df_train .index[-1]
df_price_train = df_price[:train_end_date].copy(); print('Dim. of Train dataset:', df_price_train.shape)
df_price_test = df_price[train_end_date:].copy(); print('Dim. of Test dataset:', df_price_test.shape)





# ======================================= 2. Data Analysis: Portfolio construction via ICA) =========================================================

# Number of components
k = 5
## -------------------------------------- 2.1 ICA Analysis --------------------------------------------------------------

### ICA --------------------------------------------------------------
from sklearn.decomposition import FastICA

transformer = FastICA(n_components=k, random_state=0, whiten='unit-variance')
Y_pred = transformer.fit_transform(df_train_rs)
Y_pred.shape

Y_pred = pd.DataFrame(Y_pred)
Y_pred.columns= ['IC'+str(i+1) for i in range(Y_pred.shape[1])]
Y_pred.set_index(df_train_rs.index, inplace=True)

print('Check if Means of ICs (all Means should be centored at 0 under ICA assumption):\n',Y_pred.mean(axis=0) )
print('Check SDs of ICs (all variances/SDs are constrained the same under ICA whiteing assumption):\n',Y_pred.std(axis=0) )


'''
## Rescale Y_pred s.t. rest of Var(Y_pred) have same scalse as the first IC (SD=1); actually, since each IC has equal SD, such operation is same as rescaling individual IC themselves
Y_pred_rs = (Y_pred -Y_pred.mean(axis=0)[0] )/Y_pred.std(axis=0)[0]
print('Check if Means of ICs (all Means should be centored at 0 under ICA assumption):\n',Y_pred_rs.mean(axis=0) )
print('Check if SDs of ICs are 1):\n',Y_pred_rs.std(axis=0) )


### Visualization: TS for ICs ---------------------------------------------------------------
Y_pred.plot(figsize=(15,6))
plt.legend(fancybox=True, framealpha=0.0, loc = 'upper right', prop={'size': 8})
plt.title('Daily Portforlio Returns of ICs')
plt.show()
plt.savefig(output_dir+f'/Portfolio Returns of {k} ICs ({int(df.shape[1])}D).png', transparent = True)
plt.close()

Y_pred_rs.plot(figsize=(15,6))
plt.legend(fancybox=True, framealpha=0.0, loc = 'upper right', prop={'size': 8})
plt.title('Daily Portforlio Returns of ICs (Standardized w.r.t. 1st IC))')
plt.ylim((-8,8))
plt.show()
plt.savefig(output_dir+f'/Portfolio Returns of {k} Standardized ICs ({int(df.shape[1])}D).png', transparent = True)
plt.close()
'''

## -------------------------------------- 2.2 Portforlio Weights (per IC) --------------------------------------------------------------

### Derive weight matrix and re-scaling: W (weight matrix to transform raw data X into ICs Y) ---------------------------------------------------------------
# Investigate W
W_IC  = pd.DataFrame(transformer.components_, columns=df.columns)
W_IC.set_index(Y_pred.columns, inplace=True)
#W_IC.to_csv(output_dir+f'/W matrix of {k} ICs ({int(df.shape[1])}D).csv')


## normalized each IC so that w sum up to 1
W_IC_rs = Rescale_W(W_IC)
print('Check sum of each IC:\n', W_IC_rs.sum(axis=1))


#decide threshod for truncation
k1, k2 = W_IC_rs.shape
#plt.hist(W_IC_rs.values.flatten(), bins=k1*k2)
#plt.close()
threshold = 5
## Truncate weights
W_IC_rs1 = W_IC_rs.applymap(Trunc, threshold = threshold)
W_IC_rs1 = Rescale_W(W_IC_rs1)
print('Check sum of each IC:\n', W_IC_rs1.sum(axis=1))


#  normalized each IC so that L2-norm of w sum up to 1 vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
tempt =  np.diag(np.dot(W_IC, W_IC.T))
W_IC_rs2 = W_IC.mul(1/np.sqrt(tempt), axis=0)



# Customize the heatmap oW weight matrix -------------------------------------------------------------------
W_IC_alias = W_IC_rs2 #W_IC #W_IC_rs
#vnames = [name for name in globals() if globals()[name] is W_IC_alias]
#print(vnames)

sns.set(rc={'figure.figsize':(15,4)})
sns.heatmap(W_IC_alias,  #W_IC_rs1
            annot=True,
            center = 0,
            linewidths=0.4,
            annot_kws={"size": 10})
plt.title(f'Portfolio construction with {k} ICs (weights rescaled ')
plt.xticks(rotation=45, fontsize=7)
plt.yticks(rotation=0) 
plt.show()
plt.savefig(output_dir+f'/Rescaled W matrix of {k} ICs - W_IC_rs ({int(df.shape[1])}D).png', transparent = True)
plt.close()


## -------------------------------------- 2.3 Portforlio (per IC) --------------------------------------------------------------
periods_per_year=252
W_IC_alias = W_IC_rs2 #W_IC_rs2 #W_IC #

# portfolio per IC --------------------------------------------------------------------
if df_price_train is not None:
    Y = np.dot(df_price_train, W_IC_alias.T)
    Y = pd.DataFrame(Y, columns=W_IC_alias.index, index=df_price_train.index)
    
    # Calculate returns of each ICs
    Y_returns = Y.pct_change()[1:] #daily: periods=1 for daily return
    
    # Calculate importance of each IC
    IC_w = Y_returns.agg([np.mean, np.var]) 
    IC_w =IC_w.append(pd.Series(kurtosis(Y_returns), name='kurtosis', index=W_IC_alias.index))
    
    ### Optimum Portforlio Construction ---------------------------------------------------------------
    Y_w = Y * ( np.sign(IC_w.loc['mean',:]) * (np.abs(IC_w.loc['mean',:])/IC_w.loc['kurtosis',:])**(1/3)   )
    P_ICA = Y_w.sum(axis=1)
    P_ICA_returns = P_ICA.pct_change()[1:] #daily: periods=1 for daily return
    
    '''
    Y_returns_w = Y_returns * ( np.sign(IC_w.loc['mean',:]) * (np.abs(IC_w.loc['mean',:])/IC_w.loc['kurtosis',:])**(1/3)   )
    P_ICA = Y_returns_w.sum(axis=1)
    '''
    
    ## Visualization   
    ax = Y_returns.plot(figsize=(15,6), linewidth=1.3, linestyle='dotted')   #Y_returns_w
    P_ICA_returns.plot(ax=ax, label='Optimum Portfolio', alpha=0.5, linewidth=3) #, 
    plt.legend(fancybox=True, framealpha=0.0, prop={'size': 8}) #loc = 'upper right'
    plt.title(f'Portfolio returns from {k} ICs')
    #plt.ylim((-0.2,0.4))
    plt.show()
    #                                                                         !!!!!!!
    plt.savefig(output_dir+f'/True Portfolio returns from {k} ICs with w - W_IC_rs2 ({int(df.shape[1])}D).png', transparent = True)
    plt.close()       
    

    ax = Y_returns['2019-02-12':].plot(figsize=(15,6), linewidth=1.3, linestyle='dotted')   #Y_returns_w
    P_ICA_returns['2019-02-12':].plot(ax=ax, label='Optimum Portfolio', alpha=0.5, linewidth=3) #, 
    plt.legend(fancybox=True, framealpha=0.0, prop={'size': 8}) #loc = 'upper right'
    plt.title(f'Portfolio returns from {k} ICs')
    #plt.ylim((-0.2,0.4))
    plt.show()
    #                                                                         !!!!!!!
    plt.savefig(output_dir+f'/True Portfolio returns from {k} ICs with w - W_IC_rs2 ({int(df.shape[1])}D) 2.png', transparent = True)
    plt.close() 


    ### Calculate cumprod of (Optimum ) Portforlio Returns  ---------------------------------------------------------------
    Y_returns_cumprod = np.cumprod(Y_returns + 1)
    #Y_returns_w_cumprod = np.cumprod(Y_returns_w + 1)
    P_ICA_returns_cumprod = np.cumprod(P_ICA_returns + 1)
    '''
    #                                                                         !!!!!!!
    Y_returns_cumprod.plot(title=f'Cumprod portfolio returns from {k} ICs (with W_IC)', figsize=(15,6), linewidth=3)
    plt.savefig(output_dir+f'/True Cumprod portfolio returns from {k} ICs (with W_IC_rs2).png', transparent = True)
    plt.close()
    '''
    ax = Y_returns_cumprod.plot(figsize=(15,6), linewidth=1.3)   #Y_returns_w
    P_ICA_returns_cumprod.plot(ax=ax, label='Optimum Portfolio', alpha=0.5, linewidth=3) #, 
    plt.legend(fancybox=True, framealpha=0.0, prop={'size': 8}) #loc = 'upper right'
    plt.title(f'Cumprod portfolio returns from {k} ICs')
    #plt.ylim((-0.2,0.4))
    plt.show()
    plt.savefig(output_dir+f'/True Cumprod portfolio returns from {k} ICs (with W_IC_rs2).png', transparent = True)
    plt.close()
    
    #ax = Y_returns_cumprod.plot(figsize=(15,6), linewidth=1.3)   #Y_returns_w
    P_ICA_returns_cumprod.plot() #, ax=ax, alpha=0.5, linewidth=3
    plt.legend(fancybox=True, framealpha=0.0, prop={'size': 8}) #loc = 'upper right'
    plt.title(f'Cumprod of optimum portfolio returns from {k} ICs')
    #plt.ylim((-0.2,0.4))
    plt.show()
    plt.savefig(output_dir+f'/True Cumprod optimum portfolio returns from {k} ICs (with W_IC_rs2).png', transparent = True)
    plt.close()



    ### Calculate sharpe  ---------------------------------------------------------------
    sharpe_ICA = Y_returns.apply(sharpe_ratio, periods_per_year=periods_per_year, axis=0) #index: er, vol, sharpe 
    sharpe_ICA.columns = W_IC_alias.index
    sharpe_ICA.index = ['Ann_return', 'Ann_Sharpe', 'Ann_Volatility', 'mean', 'var', 'kurtosis']
    
    sharpe_P_ICA = pd.DataFrame(sharpe_ratio(P_ICA_returns), columns=['Portfolio'], index=sharpe_ICA.index )
    sharpe_ICA = pd.concat([sharpe_ICA, sharpe_P_ICA], axis=1)
 

if df_price_test is not None:
    Y_test = np.dot(df_price_test, W_IC_alias.T)
    Y_test = pd.DataFrame(Y_test, columns=W_IC_alias.index, index=df_price_test.index)
    
    # Calculate returns of each ICs
    Y_returns_test = Y_test.pct_change()[1:] #daily: periods=1 for daily return
    


    ### Optimum Portforlio Construction ---------------------------------------------------------------
    Y_test_w = Y_test * ( np.sign(IC_w.loc['mean',:]) * (np.abs(IC_w.loc['mean',:])/IC_w.loc['kurtosis',:])**(1/3)   )
    P_ICA_test = Y_test_w.sum(axis=1)
    P_ICA_test_returns = P_ICA_test.pct_change()[1:] #daily: periods=1 for daily return
    
    '''
    Y_returns_test_w = Y_returns_test * ( np.sign(IC_w.loc['mean',:]) * (np.abs(IC_w.loc['mean',:])/IC_w.loc['kurtosis',:])**(1/3)   )
    P_ICA_test = Y_returns_test_w.sum(axis=1)
    '''
 
    ## Visualization
    ax = Y_returns_test.plot(figsize=(15,6), linewidth=1.3)   #Y_returns_w
    P_ICA_test_returns.plot(ax=ax, label='Optimum Portfolio', alpha=0.5, linewidth=3) #, 
    plt.legend(fancybox=True, framealpha=0.0, prop={'size': 8}) #loc = 'upper right'
    plt.title(f'Portfolio returns from {k} ICs')
    #plt.ylim((-0.2,0.4))
    plt.show()
    #                                                                         !!!!!!!
    plt.savefig(output_dir+f'/Test- True Portfolio returns from {k} ICs with w - W_IC_rs2 ({int(df.shape[1])}D).png', transparent = True)
    plt.close()       


    ### Calculate cumprod of (Optimum ) Portforlio Returns  ---------------------------------------------------------------
    Y_returns_test_cumprod = np.cumprod(Y_returns_test + 1)
    #Y_returns_test_w_cumprod = np.cumprod(Y_returns_test_w + 1)
    P_ICA_test_returns_cumprod = np.cumprod(P_ICA_test_returns + 1)
    '''
    #                                                                         !!!!!!!
    Y_returns_cumprod.plot(title=f'Cumprod portfolio returns from {k} ICs (with W_IC)', figsize=(15,6), linewidth=3)
    plt.savefig(output_dir+f'/Test- True Cumprod portfolio returns from {k} ICs (with W_IC_rs2).png', transparent = True)
    plt.close()
    '''
    ax = Y_returns_test_cumprod.plot(figsize=(15,6), linewidth=1.3)   #Y_returns_w
    P_ICA_test_returns_cumprod.plot(ax=ax, label='Optimum Portfolio', alpha=0.5, linewidth=3) #, 
    plt.legend(fancybox=True, framealpha=0.0, prop={'size': 8}) #loc = 'upper right'
    plt.title(f'Cumprod portfolio returns from {k} ICs')
    #plt.ylim((-0.2,0.4))
    plt.show()
    plt.savefig(output_dir+f'/Test- True Cumprod weighted portfolio returns from {k} ICs (with W_IC_rs2).png', transparent = True)
    plt.close()
    

    ### Calculate sharpe  ---------------------------------------------------------------
    sharpe_ICA_t = Y_returns_test.apply(sharpe_ratio, periods_per_year=periods_per_year, axis=0) #index: er, vol, sharpe 
    sharpe_ICA_t.columns = W_IC_alias.index
    sharpe_ICA_t.index = ['Ann_return', 'Ann_Sharpe', 'Ann_Volatility', 'mean', 'var', 'kurtosis']
    
    sharpe_P_ICA = pd.DataFrame(sharpe_ratio(P_ICA_test_returns), columns=['Portfolio'], index=sharpe_ICA_t.index )
    sharpe_ICA_t = pd.concat([sharpe_ICA_t, sharpe_P_ICA], axis=1)
  




# ======================================= 3. Data Analysis: Portfolio construction via PCA =========================================================

## -------------------------------------- 3.1 PCA Analysis --------------------------------------------------------------
### Preprocessing Data --------------------------------------------------------------
'''
# Standardize the mixed signals (i.e., mean=0, std=1)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
matrix = sc_X.fit_transform(df)


print(matrix.mean(axis=0) )
print(matrix.std(axis=0) )
'''

### PCA --------------------------------------------------------------
from sklearn.decomposition import PCA
pca = PCA(n_components=k) #n_components=None or integer
Y_pred_PCA = pca.fit_transform(df_train_rs)

'''
# rescale Y s.t. var(Yi) = 1 (in order to compare with ICA results)
sc_Y = StandardScaler()
Y_pred_PCA = sc_Y.fit_transform(Y_pred_PCA)

Y_pred_PCA.mean(axis=0) 
Y_pred_PCA.std(axis=0) 
'''

Y_pred_PCA = pd.DataFrame(Y_pred_PCA)
Y_pred_PCA.columns= ['PC'+str(i+1) for i in range(k)]
Y_pred_PCA.set_index(df_train_rs.index, inplace=True)

#Change direction of PC1 (only for better visualization, won't change results)
#Y_pred_PCA.loc[:, 'PC1'] =  -Y_pred_PCA.loc[:, 'PC1']

print('Check if Means of PCs (all Means should be centored at 0 under PCA assumption):\n',Y_pred_PCA.mean(axis=0) )
print('Check SDs of PCs (should be equivelant to eigenvalues):\n',Y_pred_PCA.std(axis=0) )

'''
## Rescale Y_pred s.t. rest of Var(Y_pred) have same scalse as the first IC (SD=1); actually, since each IC has equal SD, such operation is same as rescaling individual IC themselves
Y_pred_PCA_rs = (Y_pred_PCA -Y_pred_PCA.mean(axis=0)[0] )/Y_pred_PCA.std(axis=0)[0]
print('Check if Means of ICs (all Means should be centored at 0 under ICA assumption):\n',Y_pred_PCA_rs.mean(axis=0) )
print('Check if SDs of ICs are 1):\n',Y_pred_PCA_rs.std(axis=0) )


### Visualization: PCs ---------------------------------------------------------------
Y_pred_PCA.plot(figsize=(15,6))
plt.legend(fancybox=True, framealpha=0.0, loc = 'upper right', prop={'size': 8})
plt.title('Daily Portforlio Returns of PCs')
plt.show()
plt.savefig(output_dir+f'/Portfolio Returns of {k} PCs ({int(df.shape[1])}D).png', transparent = True)
plt.close()


Y_pred_PCA_rs.plot(figsize=(15,6))
plt.legend(fancybox=True, framealpha=0.0, loc = 'upper right', prop={'size': 8})
plt.title('Daily Portforlio Returns of PCs (Standardized w.r.t. 1st PC)')
plt.ylim((-8,8))
plt.show()
plt.savefig(output_dir+f'/Portfolio Returns of {k} Standardized PCs ({int(df.shape[1])}D).png', transparent = True)
plt.close()

'''

### Visualization: W (weight matrix to transform raw data X into ICs Y) ---------------------------------------------------------------
# Investigate W
W_PC  = pd.DataFrame(pca.components_, columns=df_train.columns) ##each row is a PC -> PC1 = pca.components_[0,:]
W_PC.set_index(Y_pred_PCA.columns, inplace=True)
#W_PC.to_csv(output_dir+f'/W matrix of {k} PCs ({int(df.shape[1])}D).csv')


#Similary, change sign of weights due fo change direction of PC1 (only for better visualization, won't change results)
#W_PC.loc['PC1', :] =  -W_PC.loc['PC1', :]


'''

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

## normalized each IC so that w sum up to 1
W_PC_rs = Rescale_W(W_PC)
print('Check sum of each PC:\n', W_PC_rs.sum(axis=1))


#decide threshod for truncation
k1, k2 = W_PC_rs.shape
#plt.hist(W_PC_rs.values.flatten(), bins=k1*k2)
#plt.close()
threshold = 5
## Truncate weights
W_PC_rs1 = W_PC_rs.applymap(Trunc, threshold = threshold)
W_PC_rs1 = Rescale_W(W_PC_rs1)
print('Check sum of each PC:\n', W_PC_rs1.sum(axis=1))


# For PCA, L2-norm of w is already constrained to 1, so no need for normalization again
tempt =  np.diag(np.dot(W_PC, W_PC.T)); print('Check L2-norm of each PC:\n', tempt)
#W_PC_rs2 = W_IC



# Customize the heatmap oW weight matrix
W_PC_alias = W_PC #W_PC_rs #W_PC_rs1

#vnames = [name for name in globals() if globals()[name] is W_IC_alias]
#print(vnames)

sns.set(rc={'figure.figsize':(15,4)})
sns.heatmap(W_PC_alias,  #W_IC_rs1
            annot=True,
            center = 0,
            linewidths=0.4,
            annot_kws={"size": 10})
plt.title(f'Portfolio construction with {k} PCs (weights rescaled)')
plt.xticks(rotation=45, fontsize=7)
plt.yticks(rotation=0) 
plt.show()
plt.savefig(output_dir+f'/Rescaled W matrix of {k} PCs - W_PC ({int(df.shape[1])}D).png', transparent = True)
plt.close()




## -------------------------------------- 2.3 Portforlio (per PC) --------------------------------------------------------------
periods_per_year=252
W_PC_alias = W_PC #W_PC_rs #W_PC_rs1

# portfolio per PC --------------------------------------------------------------------
if df_price_train is not None:
    Y_PCA = np.dot(df_price_train, W_PC_alias.T)
    Y_PCA = pd.DataFrame(Y_PCA, columns=W_PC_alias.index, index=df_price_train.index)
    
    # Calculate returns of each PCs
    Y_PCA_returns = Y_PCA.pct_change()[1:] #daily: periods=1 for daily return
    
    # Calculate importance of each PC
    PC_w = Y_PCA_returns.agg([np.mean, np.var]) 
    PC_w =PC_w.append(pd.Series(kurtosis(Y_PCA_returns), name='kurtosis', index=W_PC_alias.index))
    
    ### Optimum Portforlio Construction ---------------------------------------------------------------
    Y_PCA_w = Y_PCA * ( PC_w.loc['mean',:]/PC_w.loc['var',:]   )
    P_PCA = Y_PCA_w.sum(axis=1)
    P_PCA_returns = P_PCA.pct_change()[1:] #daily: periods=1 for daily return
    '''
    Y_PCA_returns_w = Y_PCA_returns * ( PC_w.loc['mean',:]/PC_w.loc['var',:]   )
    P_PCA = Y_PCA_returns_w.sum(axis=1)
    '''
    
    ## Visualization
    ax = Y_PCA_returns.plot(figsize=(15,6), linewidth=1.3, linestyle='dotted')   #Y_PCA_returns_w
    P_PCA_returns.plot(ax=ax, label='Optimum Portfolio', alpha=0.5, linewidth=3) #, 
    plt.legend(fancybox=True, framealpha=0.0, prop={'size': 8}) #loc = 'upper right'
    plt.title(f'Portfolio returns from {k} PCs')
    #plt.ylim((-0.2,0.4))
    plt.show()
    #                                                                         !!!!!!!
    plt.savefig(output_dir+f'/True Portfolio returns from {k} PCs with w - W_PC ({int(df.shape[1])}D).png', transparent = True)
    plt.close()       
    

    ax = Y_PCA_returns['2019-02-12':].plot(figsize=(15,6), linewidth=1.3, linestyle='dotted')   #Y_PCA_returns_w
    P_PCA_returns['2019-02-12':].plot(ax=ax, label='Optimum Portfolio', alpha=0.5, linewidth=3) #, 
    plt.legend(fancybox=True, framealpha=0.0, prop={'size': 8}) #loc = 'upper right'
    plt.title(f'Portfolio returns from {k} PCs')
    #plt.ylim((-0.2,0.4))
    plt.show()
    #                                                                         !!!!!!!
    plt.savefig(output_dir+f'/True Portfolio weighted returns from {k} PCs with w - W_PC ({int(df.shape[1])}D) 2.png', transparent = True)
    plt.close() 


    ### Calculate cumprod of (Optimum ) Portforlio Returns  ---------------------------------------------------------------
    Y_PCA_returns_cumprod = np.cumprod(Y_PCA_returns + 1)
    #Y_PCA_returns_w_cumprod = np.cumprod(Y_PCA_returns_w + 1)
    P_PCA_returns_cumprod = np.cumprod(P_PCA_returns + 1)
    '''
    #                                                                         !!!!!!!
    Y_PCA_returns_cumprod.plot(title=f'Cumprod portfolio returns from {k} PCs (with W_PC)', figsize=(15,6), linewidth=3)
    plt.savefig(output_dir+f'/True Cumprod portfolio returns from {k} PCs (with W_PC).png', transparent = True)
    plt.close()
    '''
    ax = Y_PCA_returns_cumprod.plot(figsize=(15,6), linewidth=1.3)   #Y_PCA_returns_w
    P_PCA_returns_cumprod.plot(ax=ax, label='Optimum Portfolio', alpha=0.5, linewidth=3) #, 
    plt.legend(fancybox=True, framealpha=0.0, prop={'size': 8}) #loc = 'upper right'
    plt.title(f'Cumprod portfolio returns from {k} PCs')
    #plt.ylim((-0.2,0.4))
    plt.show()
    plt.savefig(output_dir+f'/True Cumprod weighted portfolio returns from {k} PCs (with W_PC).png', transparent = True)
    plt.close()
    
    #ax = Y_PCA_returns_cumprod.plot(figsize=(15,6), linewidth=1.3)   #Y_returns_w
    P_PCA_returns_cumprod.plot() #, ax=ax, alpha=0.5, linewidth=3
    plt.legend(fancybox=True, framealpha=0.0, prop={'size': 8}) #loc = 'upper right'
    plt.title(f'Cumprod of optimum portfolio returns from {k} PCs')
    #plt.ylim((-0.2,0.4))
    plt.show()
    plt.savefig(output_dir+f'/True Cumprod optimum portfolio returns from {k} PCs (with W_PC).png', transparent = True)
    plt.close()


    ### Calculate sharpe  ---------------------------------------------------------------
    sharpe_PCA = Y_PCA_returns.apply(sharpe_ratio, periods_per_year=periods_per_year, axis=0) #index: er, vol, sharpe 
    sharpe_PCA.columns = W_PC_alias.index
    sharpe_PCA.index = ['Ann_return', 'Ann_Sharpe', 'Ann_Volatility', 'mean', 'var', 'kurtosis']
    
    sharpe_P_PCA = pd.DataFrame(sharpe_ratio(P_PCA_returns), columns=['Portfolio'], index=sharpe_PCA.index )
    sharpe_PCA = pd.concat([sharpe_PCA, sharpe_P_PCA], axis=1)



if df_price_test is not None:
    Y_PCA_test = np.dot(df_price_test, W_PC_alias.T)
    Y_PCA_test = pd.DataFrame(Y_PCA_test, columns=W_PC_alias.index, index=df_price_test.index)
    
    # Calculate returns of each PCs
    Y_PCA_returns_test = Y_PCA_test.pct_change()[1:] #daily: periods=1 for daily return
    
    
    ### Optimum Portforlio Construction ---------------------------------------------------------------
    ### Optimum Portforlio Construction ---------------------------------------------------------------
    Y_PCA_test_w = Y_PCA_test * ( PC_w.loc['mean',:]/PC_w.loc['var',:]   )
    P_PCA_test = Y_PCA_test_w.sum(axis=1)
    P_PCA_returns_test = P_PCA_test.pct_change()[1:] #daily: periods=1 for daily return    
    '''
    Y_PCA_returns_test_w = Y_PCA_returns_test * ( PC_w.loc['mean',:]/PC_w.loc['var',:]   )
    P_PCA_test = Y_PCA_returns_test_w.sum(axis=1)
    '''
    
    ## Visualization
    ax = Y_PCA_returns_test.plot(figsize=(15,6), linewidth=1.3)   #Y_PCA_returns_w
    P_PCA_returns_test.plot(ax=ax, label='Optimum Portfolio', alpha=0.5, linewidth=3) #, 
    plt.legend(fancybox=True, framealpha=0.0, prop={'size': 8}) #loc = 'upper right'
    plt.title(f'Portfolio returns from {k} PCs')
    #plt.ylim((-0.2,0.4))
    plt.show()
    #                                                                         !!!!!!!
    plt.savefig(output_dir+f'/Test- True Portfolio weighted returns from {k} PCs with w - W_PC ({int(df.shape[1])}D).png', transparent = True)
    plt.close()       


    ### Calculate cumprod of (Optimum ) Portforlio Returns  ---------------------------------------------------------------
    Y_PCA_returns_test_cumprod = np.cumprod(Y_PCA_returns_test + 1)
    #Y_PCA_returns_test_w_cumprod = np.cumprod(Y_PCA_returns_test_w + 1)
    P_PCA_returns_test_cumprod = np.cumprod(P_PCA_returns_test + 1)
    '''
    #                                                                         !!!!!!!
    Y_PCA_returns_cumprod.plot(title=f'Cumprod portfolio returns from {k} PCs (with W_PC)', figsize=(15,6), linewidth=3)
    plt.savefig(output_dir+f'/Test- True Cumprod portfolio returns from {k} PCs (with W_PC).png', transparent = True)
    plt.close()
    '''
    ax = Y_PCA_returns_test_cumprod.plot(figsize=(15,6), linewidth=1.3)   #Y_PCA_returns_w
    P_PCA_returns_test_cumprod.plot(ax=ax, label='Optimum Portfolio', alpha=0.5, linewidth=3) #, 
    plt.legend(fancybox=True, framealpha=0.0, prop={'size': 8}) #loc = 'upper right'
    plt.title(f'Cumprod portfolio returns from {k} PCs')
    #plt.ylim((-0.2,0.4))
    plt.show()
    plt.savefig(output_dir+f'/Test- True Cumprod portfolio returns from {k} PCs (with W_PC).png', transparent = True)
    plt.close()
    

    ### Calculate sharpe  ---------------------------------------------------------------
    sharpe_PCA_t = Y_PCA_returns_test.apply(sharpe_ratio, periods_per_year=periods_per_year, axis=0) #index: er, vol, sharpe 
    sharpe_PCA_t.columns = W_PC_alias.index
    sharpe_PCA_t.index = ['Ann_return', 'Ann_Sharpe', 'Ann_Volatility', 'mean', 'var', 'kurtosis']
    
    sharpe_P_PCA = pd.DataFrame(sharpe_ratio(P_PCA_returns_test), columns=['Portfolio'], index=sharpe_PCA_t.index )
    sharpe_PCA_t = pd.concat([sharpe_PCA_t, sharpe_P_PCA], axis=1)










# ======================================= 5. Summary & Visualization of Results =========================================================
        
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
fig.text(0.5, 0.90, f'Portfolio weights with first {k} PCs', ha='center', fontsize=12)

# Plot seperated signals: ICA
sns.heatmap(W_IC_rs2,
            annot=True,
            center = 0,
            linewidths=0.4,
            annot_kws={"size": 10}, ax = ax[1])
ax[1].legend(fancybox=True, framealpha=0.0, loc = 'upper right', prop={'size': 8})
fig.text(0.5, 0.48, f'Portfolio weights with first {k} ICs', ha='center', fontsize=12)


fig.text(0.5, 0.04, 'Equities', ha='center', fontsize=14, fontweight ='bold')
plt.xticks(rotation=20, fontsize=7)
plt.suptitle("Portfolio (weights) construction with PCA or ICA", fontsize=14, fontweight ='bold')
plt.savefig(output_dir+"/Compare estimates of (rescaled) W matrix from PCA-ICA.png", transparent=True)
plt.show()
plt.close()

   


### Visualization of Results from PCA/ICA --------------------------------------------------------------

## Estimation of portfolio returns from each portfolio as well as from the optimum portfolio -------------------
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(30,12)) #sharey=True,
# Plot seperated signals: PCA
ax[0].plot(Y_PCA_returns.index, Y_PCA_returns, label=Y_PCA_returns.columns, linewidth=1.3, linestyle='dotted') #
ax[0].plot(P_PCA_returns.index, P_PCA_returns, label='Optimum Portfolio', alpha=0.5, linewidth=3)
ax[0].legend(fancybox=True, framealpha=0.0, loc = 'upper center', ncol=len(Y_returns.columns)+1, prop={'size': 8})
fig.text(0.5, 0.90, f'Portfolio returns from {k} PCs', ha='center', fontsize=12)

# Plot seperated signals: ICA
ax[1].plot(Y_returns.index, Y_returns, label=Y_returns.columns, linewidth=1.3, linestyle='dotted') #
ax[1].plot(P_ICA_returns.index, P_ICA_returns, label='Optimum Portfolio', alpha=0.5, linewidth=3)
ax[1].legend(fancybox=True, framealpha=0.0, loc = 'upper center', ncol=len(Y_returns.columns)+1, prop={'size': 8})
fig.text(0.5, 0.48, f'Portfolio returns from {k} ICs', ha='center', fontsize=12)

fig.text(0.5, 0.04, 'Date', ha='center', fontsize=14, fontweight ='bold')
fig.text(0.04, 0.5, 'Portfolio Returns', va='center', rotation='vertical', fontsize=14, fontweight ='bold')
#plt.suptitle("A comparision of portfolio returns by PCA and ICA", fontsize=14, fontweight ='bold')

plt.savefig(output_dir+"/Compare estimates of portfolio returns from PCA-ICA.png", transparent=True)
plt.show()
plt.close()



fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(30,12)) #
# Plot seperated signals: PCA
ax[0].plot(Y_PCA_returns['2019-02-12':].index, Y_PCA_returns['2019-02-12':], label=Y_PCA_returns.columns, linewidth=1.3, linestyle='dotted') #
ax[0].plot(P_PCA_returns['2019-02-12':].index, P_PCA_returns['2019-02-12':], label='Optimum Portfolio', alpha=0.5, linewidth=3)
ax[0].legend(fancybox=True, framealpha=0.0, loc = 'upper center', ncol=len(Y_returns.columns)+1, prop={'size': 8})
#ax[0]
fig.text(0.5, 0.90, f'Portfolio returns from {k} PCs', ha='center', fontsize=12)

# Plot seperated signals: ICA
ax[1].plot(Y_returns['2019-02-12':].index, Y_returns['2019-02-12':], label=Y_returns.columns, linewidth=1.3, linestyle='dotted') #
ax[1].plot(P_ICA_returns['2019-02-12':].index, P_ICA_returns['2019-02-12':], label='Optimum Portfolio', alpha=0.5, linewidth=3)
ax[1].legend(fancybox=True, framealpha=0.0, loc = 'upper center', ncol=len(Y_returns.columns)+1, prop={'size': 8})
fig.text(0.5, 0.48, f'Portfolio returns from {k} ICs', ha='center', fontsize=12)

fig.text(0.5, 0.04, 'Date', ha='center', fontsize=14, fontweight ='bold')
fig.text(0.04, 0.5, 'Portfolio Returns', va='center', rotation='vertical', fontsize=14, fontweight ='bold')
#plt.suptitle("A comparision of portfolio returns by PCA and ICA", fontsize=14, fontweight ='bold')

plt.savefig(output_dir+"/Compare estimates of portfolio returns from PCA-ICA 2.png", transparent=True)
plt.show()
plt.close()

#------------------------------------------------------------------------------------------------------------------
## Estimation of cumulative returns from each portfolio as well as from the optimum portfolio -------------------
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(30,12)) #, sharey=True
# Plot seperated signals: PCA
ax[0].plot(Y_PCA_returns_cumprod.index, Y_PCA_returns_cumprod, label=Y_PCA_returns_cumprod.columns, linewidth=1.3, linestyle='dotted') #
ax[0].plot(P_PCA_returns_cumprod.index, P_PCA_returns_cumprod, label='Optimum Portfolio', alpha=0.5, linewidth=3)
ax[0].legend(fancybox=True, framealpha=0.0, loc = 'upper center', ncol=len(Y_PCA_returns_cumprod.columns)+1, prop={'size': 8})
#ax[0]
fig.text(0.5, 0.90, f'Cumprod portfolio returns from {k} PCs', ha='center', fontsize=12)

# Plot seperated signals: ICA
ax[1].plot(Y_returns_cumprod.index, Y_returns_cumprod, label=Y_returns_cumprod.columns, linewidth=1.3, linestyle='dotted') #
ax[1].plot(P_ICA_returns_cumprod.index, P_ICA_returns_cumprod, label='Optimum Portfolio', alpha=0.5, linewidth=3)
ax[1].legend(fancybox=True, framealpha=0.0, loc = 'upper center', ncol=len(Y_returns_cumprod.columns)+1, prop={'size': 8})
fig.text(0.5, 0.48, f'Cumprod portfolio returns from {k} ICs', ha='center', fontsize=12)

fig.text(0.5, 0.04, 'Date', ha='center', fontsize=14, fontweight ='bold')
fig.text(0.04, 0.5, 'Cumulative Portfolio Returns', va='center', rotation='vertical', fontsize=14, fontweight ='bold')
#plt.suptitle("A comparision of portfolio returns by PCA and ICA", fontsize=14, fontweight ='bold')

plt.savefig(output_dir+"/Compare cumulative returns from PCA-ICA.png", transparent=True)
plt.show()
plt.close()



## Estimation of cumulative returns from each portfolio as well as from the optimum portfolio -------------------
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(30,12)) #, sharey=True
# Plot seperated signals: PCA
ax[0].plot(Y_PCA_returns_cumprod.index, Y_PCA_returns_cumprod, label=Y_PCA_returns_cumprod.columns, linewidth=1.3, linestyle='dotted') #
ax[0].plot(P_PCA_returns_cumprod.index, P_PCA_returns_cumprod, label='Optimum Portfolio', alpha=0.5, linewidth=3)
ax[0].legend(fancybox=True, framealpha=0.0, loc = 'upper center', ncol=len(Y_PCA_returns_cumprod.columns)+1, prop={'size': 8})
#ax[0]
fig.text(0.5, 0.90, f'Cumprod portfolio returns from {k} PCs', ha='center', fontsize=12)

# Plot seperated signals: ICA
ax[1].plot(Y_returns_cumprod.iloc[:,1:].index, Y_returns_cumprod.iloc[:,1:], label=Y_returns_cumprod.iloc[:,1:].columns, linewidth=1.3, linestyle='dotted') #
ax[1].plot(P_ICA_returns_cumprod.index, P_ICA_returns_cumprod, label='Optimum Portfolio', alpha=0.5, linewidth=3)
ax[1].legend(fancybox=True, framealpha=0.0, loc = 'upper center', ncol=len(Y_returns_cumprod.columns)+1, prop={'size': 8})
fig.text(0.5, 0.48, f'Cumprod portfolio returns from {k} ICs', ha='center', fontsize=12)

fig.text(0.5, 0.04, 'Date', ha='center', fontsize=14, fontweight ='bold')
fig.text(0.04, 0.5, 'Cumulative Portfolio Returns', va='center', rotation='vertical', fontsize=14, fontweight ='bold')
#plt.suptitle("A comparision of portfolio returns by PCA and ICA", fontsize=14, fontweight ='bold')

plt.savefig(output_dir+"/Compare cumulative returns from PCA-ICA 2.png", transparent=True)
plt.show()
plt.close()




#  --------------------------------------  for testing sets ---------------------------------------------------------
## Estimation of portfolio returns from each portfolio as well as from the optimum portfolio -------------------
fig, ax = plt.subplots(2, 1, sharex=True,sharey=True, figsize=(30,12)) #
# Plot seperated signals: PCA
ax[0].plot(Y_PCA_returns_test.index, Y_PCA_returns_test, label=Y_PCA_returns_test.columns, linewidth=1.3, linestyle='dotted') #
ax[0].plot(P_PCA_returns_test.index, P_PCA_returns_test, label='Optimum Portfolio', alpha=0.5, linewidth=3)
ax[0].legend(fancybox=True, framealpha=0.0, loc = 'upper center', ncol=len(Y_returns.columns)+1, prop={'size': 8})
fig.text(0.5, 0.90, f'Portfolio returns from {k} PCs', ha='center', fontsize=12)

# Plot seperated signals: ICA
ax[1].plot(Y_returns_test.index, Y_returns_test, label=Y_returns_test.columns, linewidth=1.3, linestyle='dotted') #
ax[1].plot(P_ICA_test_returns.index, P_ICA_test_returns, label='Optimum Portfolio', alpha=0.5, linewidth=3)
ax[1].legend(fancybox=True, framealpha=0.0, loc = 'upper center', ncol=len(Y_returns.columns)+1, prop={'size': 8})
fig.text(0.5, 0.48, f'Portfolio returns from {k} ICs', ha='center', fontsize=12)

fig.text(0.5, 0.04, 'Date', ha='center', fontsize=14, fontweight ='bold')
fig.text(0.04, 0.5, 'Portfolio Returns', va='center', rotation='vertical', fontsize=14, fontweight ='bold')
#plt.suptitle("A comparision of portfolio returns by PCA and ICA", fontsize=14, fontweight ='bold')

plt.savefig(output_dir+"/Test- Compare estimates of portfolio returns from PCA-ICA.png", transparent=True)
plt.show()
plt.close()


## Estimation of cumulative returns from each portfolio as well as from the optimum portfolio -------------------
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(30,12)) #
# Plot seperated signals: PCA
ax[0].plot(Y_PCA_returns_test_cumprod.index, Y_PCA_returns_test_cumprod, label=Y_PCA_returns_test_cumprod.columns, linewidth=1.3, linestyle='dotted') #
ax[0].plot(P_PCA_returns_test_cumprod.index, P_PCA_returns_test_cumprod, label='Optimum Portfolio', alpha=0.5, linewidth=3)
ax[0].legend(fancybox=True, framealpha=0.0, loc = 'upper center', ncol=len(Y_PCA_returns_cumprod.columns)+1, prop={'size': 8})
#ax[0]
fig.text(0.5, 0.90, f'Cumprod portfolio returns from {k} PCs', ha='center', fontsize=12)

# Plot seperated signals: ICA
ax[1].plot(Y_returns_test_cumprod.index, Y_returns_test_cumprod, label=Y_returns_test_cumprod.columns, linewidth=1.3, linestyle='dotted') #
ax[1].plot(P_ICA_test_returns_cumprod.index, P_ICA_test_returns_cumprod, label='Optimum Portfolio', alpha=0.5, linewidth=3)
ax[1].legend(fancybox=True, framealpha=0.0, loc = 'upper center', ncol=len(Y_returns_cumprod.columns)+1, prop={'size': 8})
fig.text(0.5, 0.48, f'Cumprod portfolio returns from {k} ICs', ha='center', fontsize=12)

fig.text(0.5, 0.04, 'Date', ha='center', fontsize=14, fontweight ='bold')
fig.text(0.04, 0.5, 'Cumulative Portfolio Returns', va='center', rotation='vertical', fontsize=14, fontweight ='bold')
#plt.suptitle("A comparision of portfolio returns by PCA and ICA", fontsize=14, fontweight ='bold')

plt.savefig(output_dir+"/Test- Compare cumulative returns from PCA-ICA.png", transparent=True)
plt.show()
plt.close()

```

automatically created on 2022-08-12