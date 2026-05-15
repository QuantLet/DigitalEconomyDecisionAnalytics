#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 08:56:21 2026

@author: haerdle
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin

url = "https://www.chinadaily.com.cn/business"

headers = {"User-Agent": "Mozilla/5.0"}

r = requests.get(url, headers=headers, timeout=20)
r.raise_for_status()

html = r.text
soup = BeautifulSoup(html, "html.parser")

rows = []

for a in soup.find_all("a", href=True):
    title = a.get_text(" ", strip=True)
    link = urljoin(url, a["href"])

    # keep real China Daily business/news links only
    if len(title) < 8:
        continue
    if "chinadaily.com.cn" not in link:
        continue
    if "/business/" not in link and "/a/" not in link:
        continue
    if title.lower() in ["business", "china", "world", "culture"]:
        continue

    rows.append({
        "news_title": title,
        "news_link": link
    })

df = pd.DataFrame(rows).drop_duplicates()

print(df)
df.to_csv("ChinaDaily_business_news.csv", index=False, encoding="utf-8-sig")