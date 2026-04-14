#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:59:08 2026

@author: haerdle
"""
import pandas as pd

# Download different Treasury bill rates from FRED
# 4-Week T-bill
url_4wk = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB4WK"
df_4wk = pd.read_csv(url_4wk)
df_4wk.columns = ['DATE', '4_WEEK']

# 3-Month T-bill
url_3m = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
df_3m = pd.read_csv(url_3m)
df_3m.columns = ['DATE', '3_MONTH']

# 6-Month T-bill
url_6m = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB6"
df_6m = pd.read_csv(url_6m)
df_6m.columns = ['DATE', '6_MONTH']

# 1-Year T-bill
url_1y = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB1YR"
df_1y = pd.read_csv(url_1y)
df_1y.columns = ['DATE', '1_YEAR']

# Merge all into one dataframe
df_all = df_4wk.merge(df_3m, on='DATE', how='outer')
df_all = df_all.merge(df_6m, on='DATE', how='outer')
df_all = df_all.merge(df_1y, on='DATE', how='outer')

# Sort by date
df_all = df_all.sort_values('DATE')

# Display info
print(df_all.head())
print(df_all.tail())
print(f"\nTotal rows: {len(df_all)}")

# Save to CSV
df_all.to_csv('treasury_bill_rates.csv', index=False)
print("\nData saved to treasury_bill_rates.csv")