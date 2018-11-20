"""
Step 1: Clean SOCJobs
Step 2: Export train_data in \DerivedData
Step 3: Organize the factors/inputs (i.e. feature engineering)
Step 4: Train the Classifier Model (kNN)
Step 5: Export the trained model in .pkl using sklearn's joblib
"""
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

######################## Import Raw Dataset ########################
#seek_jobs = pd.read_csv(".\RawData\SeekJobs.csv", encoding = "ISO-8859-1").head(n=10000) # Only limit to first 10000 data because of Computer memory issues.
soc_jobs = pd.read_csv(".\RawData\SOCJobs.csv", encoding = "ISO-8859-1") # This is the dataset we would use to train the classification model.
soc_reduced = pd.read_csv(".\RawData\SOCJobs_Reduced_Set.csv", encoding = "ISO-8859-1")

######################## Data Pre-Processing and Cleaning ########################
# Remove the colons on the Column Names of SOCJobs dataset
soc_jobs.columns = [colname.replace(":","") for colname in soc_jobs.columns]

# Add column soc_reduced["Useful Categories"] into soc_jobs
soc_jobs["Useful Categories"] = soc_reduced["Useful Categories"]

# Filter soc_jobs where soc_jobs["Useful Categories"] = y
soc_jobs = soc_jobs.loc[soc_jobs['Useful Categories'] == 'y']
soc_jobs = soc_jobs.reset_index(drop=True)

# Clean column soc_jobs["Jobs related to this code"] by separating all "meaningful" words.
soc_jobs["Jobs related to this code"] = soc_jobs["Jobs related to this code"].apply(lambda text: clean_separation(text))

# Clean column soc_jobs["This unit group is part of"] by separating all "meaningful" words.
# e.g. PROFESSIONALSSub-Major -> PROFESSIONALS Sub Major
# e.g. PROFESSIONALSMajor -> PROFESSIONALS Major
soc_jobs["This unit group is part of"] = soc_jobs["This unit group is part of"].apply(lambda text: clean_separation(text))

# Clean SOCJobs["Tasks required by this job include"] by replacing ";" with ", "
soc_jobs["Tasks required by this job include"] = \
soc_jobs["Tasks required by this job include"].\
apply(lambda x:x.replace(";",", "))

# Concatenate all the relevant columns from soc_jobs and store the value in a new column, call it X.
soc_jobs["X"] = \
soc_jobs["Entry requirements of this job"]+\
" "+\
soc_jobs["Job description"]+\
" "+\
soc_jobs["Jobs related to this code"]+\
" "+\
soc_jobs["Tasks required by this job include"]+\
" "+\
soc_jobs["This unit group is part of"]
" "+\
soc_jobs["jobTitle"]

#soc_jobs.to_csv(".\DerivedData\CleanSOCjobs.csv", index=False)

# Introduct new dataframe only containing X and jobTitle columns. Which will be use for training Text Classification Model. SOCJobs
train_data = soc_jobs[["X", "jobTitle"]]
#train_data.to_csv(".\DerivedData\SOC_ML_TestData.csv", index=False)

# Transform all texts in train_data["X"] into lowercase
train_data["X"] = train_data["X"].apply(lambda x:x.lower())

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

#Testing
#print(train_data["X"][50])
#print("=====================================================================================")
#print(text_preprocess(train_data["X"][50]))

# Apply text_preprocess function to train_data["X"]
train_data["X"] = train_data["X"].apply(lambda x:text_preprocess(x))
train_data.to_csv(".\DerivedData\TrainData.csv", index=False)

######################## Vectorize train_data ########################
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus_train = [text for text in train_data["X"]]
y_train = list(train_data["jobTitle"])
# CountVectorizer
count_vectorizer = CountVectorizer()
X_train_countvec = count_vectorizer.fit_transform(corpus_train)
#print(count_vectorizer.get_feature_names()) # The features will be all the unique words
#print(X_train_countvec.shape) # The shape of the new 
#print(X_train_countvec.toarray()) # Values
#print(type(X_train_countvec))

# TD_IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(analyzer='word')
X_train_tfidf = tfidf_vectorizer.fit_transform(corpus_train)
#print(tfidf_vectorizer.get_feature_names())
#print(X_train_tfidf.shape)
#x = X_train_tfidf.toarray()

# Create DataFrames for CountVectorizer and TD_IDF Vectorizer
df_train_countvectorizer = pd.DataFrame(X_train_countvec.todense(), 
                                        columns=count_vectorizer.get_feature_names(),
                                        index=y_train)

df_train_tfidfvectorizer = pd.DataFrame(X_train_tfidf.todense(),
                                        columns=tfidf_vectorizer.get_feature_names(),
                                        index=y_train)

train_list_countvectorizer = df_train_countvectorizer.copy()
train_list_countvectorizer = train_list_countvectorizer.to_dict('index')
train_list_countvectorizer = [(train_list_countvectorizer[k], k) for k, v in train_list_countvectorizer.items()]

######################## Training ########################
"""
nltk.usage(nltk.classify.ClassifierI)
ClassifierI supports the following operations:
  - self.classify(featureset)
  - self.classify_many(featuresets)
  - self.labels()
  - self.prob_classify(featureset)
  - self.prob_classify_many(featuresets)
"""

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.neighbors import KNeighborsClassifier
#classifier = nltk.classify.NaiveBayesClassifier.train(train_list_countvectorizer) # Test with Naive Bayes classifier model
#prediction = classifier.classify_many(test_list_countvectorizer)

# Model Fitting (aka Train) and Test
classifier = SklearnClassifier(KNeighborsClassifier()).train(train_list_countvectorizer)

# Saving the trained model
from sklearn.externals import joblib
joblib.dump(classifier, "./job_classifier.pkl", compress=9)
#classifier2 = joblib.load("job_classifier.pkl")






