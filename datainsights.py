from .preprocessing import df_sampled
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

''' Sentiment Class Distribution '''

sns.countplot(data=df_sampled, x='sentiment', palette='Set2')
plt.title("Sentiment Class Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()


""" Top 20 Most Frequent Words """

from collections import Counter

all_words = " ".join(df_sampled['cleaned_text']).split()
word_freq = Counter(all_words)
common_words = word_freq.most_common(20)

import pandas as pd
freq_df = pd.DataFrame(common_words, columns=['word', 'count'])

sns.barplot(data=freq_df, x='count', y='word', palette='Blues_d')
plt.title("Top 20 Most Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.show()

""" Distribution of Review Lengths (in Words) """

df_sampled['text_length'] = df_sampled['cleaned_text'].apply(lambda x: len(x.split()))

sns.histplot(df_sampled['text_length'], bins=30, kde=True, color='purple')
plt.title("Distribution of Review Lengths (in Words)")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()


''' Keyword Frequency in Positive vs Negative Reviews '''

keywords = ['management', 'salary', 'workload', 'culture', 'benefits','life','time','culture','balance']

def keyword_freq(df, keyword):
    return df['cleaned_text'].str.contains(keyword).groupby(df['sentiment']).sum()

freq_df = pd.DataFrame({kw: keyword_freq(df_sampled, kw) for kw in keywords}).T
freq_df.columns = ['Negative', 'Positive']

freq_df.plot(kind='bar', figsize=(10,6), color=['red', 'green'])
plt.title("Keyword Frequency in Positive vs Negative Reviews")
plt.ylabel("Number of Reviews Containing Keyword")
plt.xticks(rotation=45)
plt.show()