"""
DEDA Unit 3
Introduction of Web Scraping in Python
Authors: WK Härdle Xiaorui ZUO
Date: 20240904
"""


"""
Reading XML Data Online
"""
"""
We use json module to unpack data from webpage. 
"""
import requests
import xml.dom.minidom
response = requests.get("https://home.treasury.gov/sites/default/files/interest-rates/daily_treas_bill_rates.xml")
content = response.content
"""
Now let us have a look at this web page content  
"""
print(content)

"""
One more time, this time on a google site  
"""
dataDOM = xml.dom.minidom.parseString(content)
response = requests.get("https://news.google.com/news/rss/headlines/section/q/finance%20news/finance%20news?ned=us&hl=en")
content = response.content

print(content)

"""
Reading JSON Data Online
"""
"""
We use json module to unpack US GOV data and receive the number of US passport applications each fiscal year from https://cadatacatalog.state.gov/dataset .
"""
import requests
import json
import pandas as pd
url = 'https://cadatacatalog.state.gov/dataset/a765ec3a-cf98-4722-a562-40c3f03d24d5/resource/a3bb04a8-dcda-4a03-ba87-e4ec63f2c4c3/download/passportapplicationbyfiscalyear.json'
response = requests.get(url)
content = json.loads(response.content)
Year = [item['Year'] for item in content]
Count = [item['Count'] for item in content]
Info = zip(Year, Count)
PassportData = pd.DataFrame(list(Info), columns=['Year', 'Count'])

print(PassportData)


"""
Webpage with RSS Feed
"""
"""
Retrieving the titles from the Financial Times Website (https://www.ft.com/?edition=international) 
"""
!pip install feedparser
import feedparser
news = feedparser.parse("https://www.ft.com/?edition=international&format=rss")

for index, item in enumerate(news.entries): # list all titles
    print("{0}.{1}".format(index, item["title"]))
    
    
