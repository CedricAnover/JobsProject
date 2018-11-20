"""
Step 1: Load the trained model
Step 2: 
"""
from sklearn.externals import joblib
###############################################################################
import os, sys, re, itertools
import pandas as pd
import sklearn as sk
import nltk, textblob, spacy
#import wordninja
import string
from nltk import word_tokenize
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

from spacy.lang.en.stop_words import STOP_WORDS
import en_core_web_sm

from auxi import clean_separation
# https://spacy.io/usage/models#usage-import import as a user instead

nlp = en_core_web_sm.load() # Language Model, english
stop_words = list(STOP_WORDS) # Stopwords
punctuations = string.punctuation # All the punctuations. To be removed.
###############################################################################
seek_jobs = pd.read_csv(".\RawData\SeekJobs.csv", encoding = "ISO-8859-1").head(n=10000)

# Concatenate all the relevant columns from seek_jobs and store the value in a new column, call it X.
seek_jobs["X"] = \
seek_jobs["Field"]+\
" "+\
seek_jobs["Department"]+\
" "+\
seek_jobs["JobTitle"]+\
" "+\
seek_jobs["JobDescription"]

# Introduct new dataframe only containing X and jobTitle columns. Which will be use for testing Text Classification Model. SEEKJobs
test_data = seek_jobs[["X", "JobTitle"]]
test_data["Job Category"] = np.NaN

# Transform all texts in test_data["X"] into lowercase
test_data["X"] = test_data["X"].apply(lambda x:x.lower())

### Preprocess train_data["X"] with NLP tools
# Tokenize, Lemmatize, Stemming, Stop words removal, Punctuation removal, etc.
# e.g. "Cedric programming programmed. Cedratic is Happy, and Lucky! Wh's this?"
# --> "Cedric program program cedric happy lucky what this"
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
def text_preprocess(text):
    """Pipeline for Cleaning text with NLP tools"""
    ### Tokenize
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)
    ### Lemmatize and Stemming
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    ### Remove Stop words
    newtokens = []
    for token in [token.lower() for token in tokens]:
        if token not in stop_words:
            newtokens.append(token)
            
    ### Remove Punctuations
    newtokens2 = []
    for token in newtokens:
        if token not in punctuations:
            newtokens2.append(token)
    
    ### Remove punctuations concatenated with words
    def remove_punc_words(t): #e.g. "Cedric."->"Cedric"
        return "".join((char for char in t if char not in string.punctuation))
    
    newtokens3 = \
    map(remove_punc_words,newtokens2)

    ### Remove digits/numbers concatenated with words
    
    ### Remove single alphabets and numbers
    newtokens4 = [x for x in newtokens3 if x.isdigit()==False]
    
    # Return a clean text. Not a list of strings!!!
    return ' '.join(newtokens4)

######################## Vectorize test_data Vectorize train_data ########################
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# test_data ['X', 'JobTitle', 'Job Category']
corpus_test = [text for text in test_data["X"]]
# CountVectorizer
count_vectorizer = CountVectorizer()
X_test_countvec = count_vectorizer.fit_transform(corpus_test)

# TD_IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(analyzer='word')
X_test_tfidf = tfidf_vectorizer.fit_transform(corpus_test)
# Create DataFrames for CountVectorizer and TD_IDF Vectorizer
df_test_countvectorizer = pd.DataFrame(X_test_countvec.todense(), 
                                        columns=count_vectorizer.get_feature_names())

df_test_tfidfvectorizer = pd.DataFrame(X_test_tfidf.todense(),
                                        columns=tfidf_vectorizer.get_feature_names())


test_list_countvectorizer = df_test_countvectorizer.copy()
test_list_countvectorizer = test_list_countvectorizer.to_dict('index')
test_list_countvectorizer = [test_list_countvectorizer[k] for k, _ in test_list_countvectorizer.items()]

###############################################################################

# Load the trained model
classifier = joblib.load("job_classifier.pkl")

prediction = classifier.classify_many(test_list_countvectorizer)
test_data["Job Category"] = pd.Series(prediction)
test_data.to_csv(".\DerivedData\TestData.csv", index=False)


