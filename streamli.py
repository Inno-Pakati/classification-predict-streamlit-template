#import stremlit dependencies
import streamlit as st
import joblib, os

#import data dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import text dependencies
import re
from string import punctuation
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter

#import other dependencies
import nltk
import seaborn as sns

#define funtions
def noiseremoval(data):

    #remove urls
    data = re.sub(r'https://t.co/\w+', '', data).strip()

    # Remove puctuation
    punctuation = re.compile("[.;:!\'’‘“”?,\"()\[\]]")
    tweet = punctuation.sub("", data.lower()) 