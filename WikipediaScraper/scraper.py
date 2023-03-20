import requests
from bs4 import BeautifulSoup
import re

url = 'https://en.wikipedia.org/wiki/Chess' #Special:Random

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Get the title
title = soup.find(id='firstHeading').text
print(title)
# Get all the paragraphs
paragraphs = soup.find_all('p')

article = []
article.append(title)
for paragraph in paragraphs:
    article.append(paragraph.text)



#print(article)
saved_path = f'.//WikipediaScraper/data/{title}.txt'
with open(saved_path, 'w+') as f:
    for line in article:
        line = re.sub(r'\[[0-9]*\]', ' ', line)
        f.write(line)
