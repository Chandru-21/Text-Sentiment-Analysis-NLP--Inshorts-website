OBJECTIVE:

Predicting the sentiment of news headline and classifying it to proper news category (web scraped data from Inshorts website) 

WEBSCRAPING:

Here I used BeautifulSoup library for Scraping
I scraped data from the below html pages,
'https://inshorts.com/en/read/technology.html',
'https://inshorts.com/en/read/sports.html',
'https://inshorts.com/en/read/world.html'

PREPROCESSING:

Using NLTK library I did the following things,
Removing Stopwords-Stopwords are the words in any language which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence eg-“the”, “a”, “an”, “in"

Stemming,Lemmatization-Reduce a word to its root or base unit eg:eating, eats, eaten root verb is eat.

Removing accented characters-eg-Café,Naïve

Expanding contractions-eg-Could've to could have

Removing unwanted characters using RE -eg removing emoticons,url,html tags etc..

UNSUPERVISED LEARNING:

Using affin,vader,textblob predicting the sentiment of the headline,whether they are of Postive Sentiment or Negative Sentiment

SUPERVISED LEARNING:

Using Bagofwords Countvectorizer method we put each word in a sentence to a bag and then feed it to classifcation algorithms(logistic,Naivebayes,Decision tree).
The classification is done to classify the news to proper category.

EVALAUTION METRIC:

Confusion matrix,Accuracy score

