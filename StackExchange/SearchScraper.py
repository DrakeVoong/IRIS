from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import time


url = "https://stackoverflow.com/questions?tab=frequent&page=1"
base_url = "https://stackoverflow.com/questions"
PAGE_LIMIT = 10


def build_url(community, tab, page):
    return f"https://{community}.com/questions?sort={tab}&page={page}"


def get_questions(df, url, session):
    response = session.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    titles = []
    name_tags = []
    questions = soup.find_all('div', class_='s-post-summary js-post-summary')

    if df is not None:
        existing_questions = df['title'].tolist()
    else:
        existing_questions = []
        
    for question in questions:
        name_tag = []

        title = question.find('a', class_='s-link').get('href')

        if title in existing_questions:
            continue
        
        tags = question.find_all('li', class_= 'd-inline mr4 js-post-tag-list-item')
        for tag in tags:
            name_tag.append(tag.text)

        titles.append(title)
        name_tags.append(name_tag)

    return titles, name_tags

def next_page(df, community, url, session):
    questions = []
    titles = []
    name_tags = []

    for i in range(1, PAGE_LIMIT + 1):
        print(f"Page {i} of {community}...")

        url = build_url(community, 'MostFrequent', i)
        new_titles, new_name_tags = get_questions(df, url, session)

        list_community = [community] * len(new_titles)

        new_df = pd.DataFrame(list(zip(list_community, new_titles, new_name_tags)), columns=['community', 'title', 'name_tags'])
        if os.path.isfile('.//StackExchange/questions.csv'):
            new_df.to_csv('.//StackExchange/questions.csv', mode='a', header=False, index=False)
        else:
            new_df.to_csv('.//StackExchange/questions.csv', mode='w', header=True, index=False)

        time.sleep(5)



if __name__ == '__main__':

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'})
  
    websites = ['stackoverflow', 'unix.stackexchange', 'math.stackexchange', 'askubuntu', 'superuser', 'serverfault', 'salesforce.stackexchange', 'tex.stackexchange', 
                'stats.stackexchange', 'dba.stackexchange', 'gis.stackexchange', 'electronics.stackexchange', 'apple.stackexchange', 'physics.stackexchange',
                'wordpress.stackexchange', 'english.stackexchange', 'blender.stackexchange', 'datascience.stackexchange', 'softwareengineering.stackexchange',
                'security.stackexchange', 'gaming.stackexchange', 'rpg.stackexchange', 'diy.stackexchange', 'ell.stackexchange', 'academia.stackexchange',
                'academia.stackexchange', 'chemistry.stackexchange', '']
    data = []

    if os.path.exists('.//StackExchange/questions.csv'):
        df = pd.read_csv('.//StackExchange/questions.csv')
    else:
        df = None

    for website in websites:
        next_page(df, website, url, session)