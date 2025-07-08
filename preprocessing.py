from .dataset import path 
import pandas as pd
import os

df = pd.read_csv(os.path.join(path, 'all_reviews.csv'))

df['pros'] = df['pros'].fillna('')
df['cons'] = df['cons'].fillna('')
df['review_text'] = df['pros'] + ' ' + df['cons']


df = df[df['review_text'].str.strip() != '']

def get_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating <= 2:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['rating'].apply(get_sentiment)

df = df[df['sentiment'] != 'neutral']


df_clean = df[['review_text', 'sentiment', 'job']]

df_clean = df_clean.dropna()

df_sampled = df_clean.sample(n=15000, random_state=42).reset_index(drop=True)
df_sampled.to_csv('sentiment_dataset_15k.csv', index=False)

print(df_sampled.head())

print(df_sampled.count())


import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\r\n|\n|\t', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df_sampled['cleaned_text'] = df_sampled['review_text'].apply(preprocess_text)

print(df_sampled[ 'cleaned_text'].head(10))

