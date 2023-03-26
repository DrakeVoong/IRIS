import requests
from bs4 import BeautifulSoup
import re
import threading
import pandas as pd
import os 

def save_data_csv(file_name, data):
    df = pd.DataFrame(data)
    df = df.sample(frac=1)
    if not os.path.exists(file_name):
        df.to_csv(file_name, encoding='utf-8', escapechar="|", index=False, header=True)
    else:
        df.to_csv(file_name, encoding='utf-8', escapechar="|", index=False, header=False, mode='a')


def save_data_txt(file_name, data):
    with open(file_name, 'w+') as f:
            for line in data:
                line = re.sub(r'\[[0-9]*\]', ' ', line)
                f.write(line)

def scrape(language):
    session = requests.Session()
    while True:
        url = f'https://{language}.wikipedia.org/wiki/Special:Random' #Special:Random

        #response = requests.get(url)
        response = session.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Get the title
        title = soup.find(id='firstHeading')
        if title is None:
            continue
        title = title.text

        if '/' in title:
            continue

        # Check if the page is a stub
        is_stub = soup.find(class_='stub')
        if is_stub is not None:
            continue

        # Get all the paragraphs
        paragraphs = soup.find_all('p')
        if paragraphs is None:
            continue

        article = []
        for paragraph in paragraphs:
            article.append(paragraph.text)

        article = ' '.join(article)
        article = re.sub(r'\[[0-9]*\]', ' ', article)
        #article = re.sub(r'[^\x00-\x7F\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', ' ', article)  # remove non-ASCII and non-Japanese/Chinese characters
        #article = re.sub(r'[\s&&[^\n\t]]+', ' ', article) # replace multiple spaces with a single space, but keep newlines and tabs

        print(title)
        data = {'title': [str(title)], 'article': [str(article)]}

        save_data_csv('.//WikipediaScraper/data.csv', data)



languages = ['en', 'fr', 'jp', 'es', 'zh']
threads = []

for language in languages:
    thread = threading.Thread(target=scrape, args=(language,))
    threads.append(thread)
    thread.start()