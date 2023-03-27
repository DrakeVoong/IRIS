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


def get_questions(url, session):
    response = session.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    titles = []
    name_tags = []
    questions = soup.find_all('div', class_='s-post-summary js-post-summary')

    for question in questions:
        name_tag = []

        title = question.find('a', class_='s-link').get('href')

        tags = question.find_all('li', class_= 'd-inline mr4 js-post-tag-list-item')
        for tag in tags:
            name_tag.append(tag.text)

        titles.append(title)
        name_tags.append(name_tag)

    return titles, name_tags

def next_page(community, url, session):
    questions = []
    titles = []
    name_tags = []

    for i in range(1, PAGE_LIMIT + 1):
        print(f"Page {i} of {community}...")

        url = build_url(community, 'MostFrequent', i)
        new_titles, new_name_tags = get_questions(url, session)
        titles.extend(new_titles)
        name_tags.extend(new_name_tags)
        time.sleep(2)

    communities = [community]*len(titles)
    return communities, titles, name_tags

if __name__ == '__main__':

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'})
  
    websites = ['math.stackexchange', 'physics.stackexchange', 'chemistry.stackexchange', 'biology.stackexchange', 'cs.stackexchange', 'stats.stackexchange', 'datascience.stackexchange']
    data = []

    for website in websites:
        communities, titles, name_tags = next_page(website, url, session)
        data += list(zip(communities, titles, name_tags))
    df = pd.DataFrame(data, columns=['community', 'title', 'name_tags'])
    if os.path.exists('.//StackExchange/questions.csv'):
        df.to_csv('.//StackExchange/questions.csv', mode='a', header=False, index=False)
    else:
      df.to_csv('.//StackExchange/questions.csv', index=False)
    