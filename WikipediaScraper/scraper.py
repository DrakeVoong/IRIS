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

def scrape():
    session = requests.Session()
    while True:
        url = 'https://en.wikipedia.org/wiki/Special:Random' #Special:Random

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

        print(title)
        data = {'title': [str(title)], 'article': [str(article)]}

        save_data_csv('.//WikipediaScraper/data.csv', data)

"""
        saved_path = f'.//WikipediaScraper/data/{title}.txt'
        save_data_txt(saved_path, article)
"""
# run 2 scrape() as a thread
thread1 = threading.Thread(target=scrape)
thread1.start()
thread2 = threading.Thread(target=scrape)
thread2.start()
thread3 = threading.Thread(target=scrape)
thread3.start()
thread1.join()
thread2.join()
thread3.join()