#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:48:27 2024

@author: Hua LEI, Wolfgang Karl Härdle
"""
# Load modules
import requests
from bs4 import BeautifulSoup as soup
# Receiving source code from the China Daily website
scmp_url = 'https://www.chinadaily.com.cn/business'
url_request = requests.get(scmp_url)
# Returns the content of the response
url_content = url_request.content

# Using BeautifulSoup to parse webpage source code
parsed_content = soup(url_content, 'html.parser')
print(parsed_content)
# Find all news sections
filtered_parts = parsed_content.find_all('div', class_="tBox2") 
print(filtered_parts)
page_info = []

# For loop iterates over every line in text
for section in filtered_parts:
    #print(section)
    unit_info = {}
    # (1) Filter title
    filtered_part1 = section.find_all('a', target="_blank")
    # (2) Extract the title and link from the section
    news_title = filtered_part1[0].text.strip() 
    news_link = filtered_part1[0].get('href').strip() 
    news_link = f"https:{news_link}"   # adjust the relative link
    # (3) Add all info into the dictionary
    unit_info['news_title'] = news_title
    unit_info['news_link'] = news_link
    page_info.append(unit_info)

# print(filtered_parts)

import pandas as pd
import os
# Calling DataFrame constructor on our list 
df = pd.DataFrame(page_info, columns=['news_title', 'news_link'])
print(df)

# Exporting to .csv file 
# df.to_csv('C:/Users/17842/Desktop/Chinadaily_business_Scraped_News0925.csv')
