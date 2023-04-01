from bs4 import BeautifulSoup
import requests
import pandas as pd
import time 
import os


base_url = "https://stackoverflow.com/"

def open_pd():
    df = pd.read_csv('.//StackExchange/questions.csv')
    return df

def build_url(community, title):
    return f"https://{community}.com{title}"


def get_questions(soup):
    question = soup.find('a', class_='question-hyperlink')
    if question is None:
        return None, None, None
    question = question.text
    
    contents = ''
    body = soup.find('div', class_='s-prose js-post-body')

    if body is None:
        return None, None, None

    paragraphs = body.find_all('p')
    contents = '\n'.join([p.text for p in paragraphs])
    score = int(soup.find('div', class_='js-vote-count').get('data-value'))

    return question, contents, score

def get_answer(soup):
    answer_contents = ''
    temp_scores = {}
    answer = soup.find('div', class_='answer js-answer accepted-answer js-accepted-answer')

    if answer is None:
        answers = soup.find_all('div', class_='answer js-answer')
        if len(answers) == 0:
            return None, None
        for index, answer in enumerate(answers):
            temp_scores[index] = int(answer.get('data-score'))

        answer = answers[max(temp_scores, key=temp_scores.get)]

    answer_body = answer.find('div', class_='s-prose js-post-body')
    answer_contents = ''.join([element.text for element in answer_body])
    answer_score = answer.find('div', class_='js-vote-count').get('data-value')

    return answer_contents, answer_score

def scrape_qna_post(url, session):
    print(url)

    response = session.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    question, contents, score = get_questions(soup)

    if (question is None) or (score < 10):
        return None, None, None, None, None

    answer_contents, answer_score = get_answer(soup)

    if answer_contents is None:
        return None, None, None, None, None

    return question, contents, score, answer_contents, answer_score

def iterate_df(Questions_df, session):

    for i in range(len(Questions_df)):
        community = Questions_df['community'][i]
        title = Questions_df['title'][i]
        url = build_url(community, title)
        new_questions, new_contents, new_score, new_answer_contents, new_answer_score = scrape_qna_post(url, session)
        if new_contents is None:
            continue

        df = pd.DataFrame({'question': new_questions, 'contents': new_contents, 'score': new_score, 'answer_contents': new_answer_contents, 'answer_score': new_answer_score}, index=[0])
        if os.path.isfile('.//StackExchange/qna.csv'):
            df.to_csv('.//StackExchange/qna.csv', mode='a', header=False, index=False, encoding='utf-8')
        else:
            df.to_csv('.//StackExchange/qna.csv', mode='w', header=True, index=False, encoding='utf-8')

        time.sleep(5)


if __name__ == '__main__':
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'})

    Questions_df = open_pd()
    iterate_df(Questions_df, session)