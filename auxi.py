"""
This module contains all auxilliary and supporting functions & classes
"""

import wordninja
import itertools

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import word_tokenize

__all__ = ["clean_separation"]



def clean_separation(text):
    tokens = word_tokenize(text)
    new_tokens = list(map(lambda token: wordninja.split(token),tokens)) # List of Lists
    new_tokens = list(itertools.chain.from_iterable(new_tokens))
    return ' '.join(new_tokens)




