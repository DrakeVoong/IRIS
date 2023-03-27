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

def scrape_qna_post(url, session):
    print(url)


    response = session.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    question = soup.find('a', class_='question-hyperlink')
    if question is None:
        return None, None, None, None, None
    question = question.text
    
    contents = ''
    body = soup.find('div', class_='s-prose js-post-body')

    if body is None:
        return None, None, None, None, None

    paragraphs = body.find_all('p')
    contents = '\n'.join([p.text for p in paragraphs])
    score = soup.find('div', class_='js-vote-count').get('data-value')

    answer_contents = ''
    accepted_answer = soup.find('div', class_='answer js-answer accepted-answer js-accepted-answer')

    if accepted_answer is None:
        return None, None, None, None, None
    
    answer_body = accepted_answer.find('div', class_='s-prose js-post-body')
    answer_contents = '\n'.join([element.text for element in answer_body])
    answer_score = accepted_answer.find('div', class_='js-vote-count').get('data-value')   
    return question, contents, score, answer_contents, answer_score

def iterate_df(Questions_df, session):
    questions = []
    contents = []
    score = []
    answer_contents = []
    answer_score = []

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
        """
        questions.append(new_questions)
        contents.append(new_contents)
        score.append(new_score)
        answer_contents.append(new_answer_contents)
        answer_score.append(new_answer_score)
        """
        time.sleep(2)

    #return questions, contents, score, answer_contents, answer_score
if __name__ == '__main__':
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'})

    Questions_df = open_pd()
    iterate_df(Questions_df, session)