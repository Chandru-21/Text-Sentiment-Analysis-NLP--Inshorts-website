# -*- coding: utf-8 -*-
"""
Created on Sat May  2 11:35:00 2020

@author: Chandra mouli
"""


import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

seed_urls = ['https://inshorts.com/en/read/technology',
             'https://inshorts.com/en/read/sports',
             'https://inshorts.com/en/read/world']
def build_dataset(seed_urls):
    news_data = []
    for url in seed_urls:
        news_category = url.split('/')[-1]
        data = requests.get(url)
        soup = BeautifulSoup(data.content, 'html.parser')
        
        news_articles = [{'news_headline': headline.find('span', 
                                                         attrs={"itemprop": "headline"}).string,
                          'news_article': article.find('div', 
                                                       attrs={"itemprop": "articleBody"}).string,
                          'news_category': news_category}
                         
                            for headline, article in 
                             zip(soup.find_all('div', 
                                               class_=["news-card-title news-right-box"]),
                                 soup.find_all('div', 
                                               class_=["news-card-content news-right-box"]))
                        ]
        news_data.extend(news_articles)
    df =  pd.DataFrame(news_data)
    df = df[['news_headline', 'news_article', 'news_category']]
    return df


news_df = build_dataset(seed_urls)
news_df.head(10)
print(news_df)
c=news_df['news_headline']
print(c)

import nltk
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

from bs4 import BeautifulSoup
import numpy as np
import re
import tqdm
import unicodedata

#EDA

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text
import re
contractions_dict = {
    'didn\'t': 'did not',
    'don\'t': 'do not',
    "aren't": "are not",
    "can't": "cannot",
    "cant": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "didnt": "did not",
    "doesn't": "does not",
    "doesnt": "does not",
    "don't": "do not",
    "dont" : "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i had",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they had",
    "they'd've": "they would have",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
    }

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


import tqdm

def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

def remove_stopwords(text, is_lower_case=False, stopwords=None):
    if not stopwords:
        stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def pre_process_corpus(docs):
    norm_docs = []
    for doc in tqdm.tqdm(docs):
        doc = strip_html_tags(doc)
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = doc.lower()
        doc = remove_accented_chars(doc)
        doc = expand_contractions(doc)
        doc=remove_stopwords(doc)
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()  
        norm_docs.append(doc)
  
    return norm_docs


x= pre_process_corpus(c)
x1=pd.DataFrame(x)
x1.columns=['text']




from nltk.corpus import opinion_lexicon
pos_list=set(opinion_lexicon.positive())
neg_list=set(opinion_lexicon.negative())
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from afinn import Afinn

#using textblob
import textblob

def score(text):
    from textblob import TextBlob
    return TextBlob(text).sentiment.polarity
def predict(text):
    x1['score']=x1['text'].apply(score)
    return(x1)
    
x2=predict(x1)    
x2['Sentiment']=['positive' if score >=0 else 'negative' for score in x2['score']]
news_headline= np.array(x2['text'])
sentiments = np.array(x2['Sentiment'])

## Evaluation of performance#cannot be done
afn = Afinn(emoticons = True)
afn.score("I love it")
x_afinn=pd.DataFrame(x)
x_afinn.columns=['text']
def score(text):
    from afinn import Afinn
    return afn.score(text)
def predict(text):
    x_afinn['score']=x_afinn['text'].apply(score)
    return(x_afinn)
x1_afinn=predict(x_afinn)
x1_afinn['Sentiment']=['positive' if score >=0 else 'negative' for score in x1_afinn['score']]

#sentiment analyzing using vader model
x_vader=pd.DataFrame(x)
x_vader.columns=['text']
def score(text):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    vader=SentimentIntensityAnalyzer()
    return vader.polarity_scores(text)['compound']
def predict(text):
    x_vader['score']=x_vader['text'].apply(score)
    return(x_vader)
x1_vader=predict(x_vader)
x1_vader['Sentiment']=['positive' if scores>=0 else 'negative' for scores in x1_vader['score']]
##############################################################################################
###supervised and unsupervised sentiment  classification

reviews = x1_vader['text']
sentiment = x1_vader['Sentiment']
train_reviews=reviews.iloc[:50]
test_reviews=reviews.iloc[50:]
train_sentiment=sentiment.iloc[:50]
test_sentiment=sentiment.iloc[50:]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(reviews,sentiment,test_size=1/3,random_state=0)

#supervised Bag of words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(binary = False, min_df = 1,max_df = 5, ngram_range=(1,2))
cv_train_features = cv.fit_transform(x_train)
cv_test_features = cv.transform(x_test)
cv_test_features.shape


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty = 'l2',max_iter = 500,C= 1,solver = 'lbfgs')

lr.fit(cv_train_features,y_train)
lr_predictions = lr.predict(cv_test_features )
lr_predictions


from sklearn.metrics import accuracy_score
accuracy_score(y_test,lr_predictions)

#tfid
from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(use_idf = True,min_df = 1,max_df = 5,ngram_range=(1,2))
tv_train_features = tv.fit_transform(x_train)
tv_test_features = tv.transform(x_test)

lr.fit(tv_train_features,y_train)  

lr_predictions = lr.predict(tv_test_features)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,lr_predictions)

from sklearn.metrics import confusion_matrix, classification_report

labels = ['negative','positive']
print(classification_report(y_test,lr_predictions))


labels = ['negative','positive']
pd.DataFrame(confusion_matrix(y_test,lr_predictions),index = labels,columns = labels)



    


