import configparser
import pandas as pd
import praw
import json
import csv
import os
import re


class RedditBot:
    def __init__(self) -> None:
        # Get file path
        self.path = os.path.realpath(__file__)
        self.main_dir = os.path.dirname(self.path)

        # Subreddits to scrape
        self.subreddit_list = ['askreddit', 'nostupidquestions', 'casualconversation', 'askscience', 'askwomen', 'askmen']

        # Set comment limit per submission
        self.comments_limit = 1000

        # Initialize an instance of reddit class from PRAW
        # Grab account credentials
        config = configparser.ConfigParser()
        config.read(".//config.ini")
        print(config.sections())

        client_id = 'wUUB9pZ2ukhl_w6Yg77IYw'
        print(config)
        secret_key = config['reddit']['secret']
        password = config['reddit']['password']

        os.chdir(self.main_dir)

        # Setup
        self.reddit = praw.Reddit(client_id=client_id,
                                  client_secret=secret_key,
                                  username='Bananasplit40404',
                                  password=password,
                                  user_agent='MyAPI',
                                  ratelimit_seconds=600)

    def get_submission(self, subreddit, time_filter='month', sub_limit=10):
        """Grab every comments in a submission from a subreddit"""
        subreddit_data = self.reddit.subreddit(subreddit)
        hot_python = subreddit_data.top(time_filter=time_filter, limit=sub_limit)

        sub, subtitles, sub_bodies, sub_comments = [], [], [], []
        for submission in hot_python:
            if not submission.stickied:
                if not submission.over_18:
                    # print(dir(submission))
                    # Clean submission title for any emojis and other languages
                    # Limit the amount of unique characters for smaller model input
                    subtitle = submission.title
                    subtitle = re.sub(r'[^\x00-\x7F]+', ' ', subtitle)
                    if len(subtitle) > 100:
                        continue
                    print(
                        f'Title: {submission.title}, num_comments: {submission.num_comments}, Ups: {submission.ups}, '
                        f'Downs: {submission.downs}, ID:{submission.id}')

                    # If comment exceeds 500, Morecomments attribute error
                    submission.comment_sort = 'best'
                    submission.comments.replace_more(limit=50, threshold=5)

                    sub_body = submission.selftext


                    try:

                        sub_comments = self.get_comments(submission)
                        sub = [f'{subreddit}'] * self.comments_limit
                        subtitles = [f'{subtitle}'] * self.comments_limit
                        sub_bodies = [f'{sub_body}'] * self.comments_limit

                        df = pd.DataFrame(list(zip(sub, subtitles, sub_bodies, sub_comments)))
                        df = df.sample(frac=1)
                        df.to_csv('new_data.csv', encoding='utf-8', escapechar="|", mode='a', index=False, header=False)
                    except AttributeError:
                        print("IDK man - MoreComments object has no attribute 'body'")
                        
        return sub, subtitles, sub_bodies, sub_comments

    def get_comments(self, submission):
        """Get every comments in a submission"""
        sub_comments = []
        comment_total = 0
        for comment in submission.comments.list():
            if (comment.id != "_") and (comment_total < self.comments_limit):
                if comment.parent_id == comment.link_id:
                    if len(comment.body) > 100:
                        continue
                    comment_total += 1
                    sub_comments.append(re.sub(r'[^\x00-\x7F]+', ' ', comment.body))
        print(f'Comment total: {comment_total}')
        return sub_comments

    def multiple_subreddit(self):
        sub, subtitles, sub_bodies, sub_comments = [], [], [], []
        for subreddit in self.subreddit_list:
            tempsub, tempsubtitles, tempsub_bodies, tempsub_comments = self.get_submission(subreddit, 'month', 3000)
            sub.extend(tempsub)
            subtitles.extend(tempsubtitles)
            sub_bodies.extend(tempsub_bodies)
            sub_comments.extend(tempsub_comments)
        return sub, subtitles, sub_bodies, sub_comments

    def convert_to_csv(self, file_name='data.csv'):
        sub, subtitles, sub_bodies, sub_comments = self.multiple_subreddit()
        df = pd.DataFrame(list(zip(sub, subtitles, sub_bodies, sub_comments)))
        df = df.sample(frac=1)
        #df.to_csv(file_name, encoding='utf-8', escapechar="|")

    def csv_to_txt(self, file_name='data.txt'):
        df = pd.read_csv(file_name, encoding='utf-8', keep_default_na=False)
        data = ""
        for _, row in df.iterrows():
            if (len(str(row[2])) > 100) or (len(str(row[4])) > 100):
                continue
            contents = '\n\n' + str(row[2]) + '\n' + str(row[4])
            contents = re.sub(r'[^\x00-\x7F]+', ' ', contents)
            data += contents

        with open('data.txt', 'w', encoding='utf-8') as f:
            f.write(data)

if __name__ == "__main__":
    bot = RedditBot()
    file_name = 'new_data.csv'
    bot.multiple_subreddit()
    