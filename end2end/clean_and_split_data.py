"""
Date: 03/09/2018
Version: 2.1
Description: 1) Cleans the Weebit preprocessed data corpus
             2) Return training and test data
Python Version: 3.6
Author: Team-1b-Deep Text Eval(Shlomi Hod, Maximilian Seidler,Vageesh Saxena)
"""


####################### Importing libraries ####################################
import re
import string

import numpy as np
import pandas as pd

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
################################################################################


########################## Defining functions ##################################
def read_data():
    # Reads the preprocessed data
    """
    return : pandas dataframe with sentences and class labels
    """
    # Loading dataset
    df = pd.read_hdf("weebit.h5","text_df")[['text','y']]
    # Dropping null values
    df.dropna(inplace=True)
    # Converting class labels to int dtype
    df['y'] = df['y'].astype(int)
    print("Shape of the dataset:",df.shape)
    return df

# Cleaning the text data
def clean_text(text):
    # Remove puncuation
    text = text.translate(string.punctuation)
    # Convert words to lower case and split them
    text = text.lower().split()
    # Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    text = " ".join(text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.split()
    # stemmer = SnowballStemmer('english')
    # stemmed_words = [stemmer.stem(word) for word in text]
    # text = " ".join(stemmed_words)
    return text

def tokenize_and_convert(df):
    # Because of the computational expenses, I am using the top 20000 unique words.
    # At first, the text is tokenized and then convert those into sequences.
    # I have kept 50 words to limit the number of words in each comment.
    """
    df: pandas dataframe(output from clean_text function)
    return: data, word_index=unique tokenizers, and labels
    """
    vocabulary_size = 20000
    # Initializing Tokenizer from keras
    tokenizer = Tokenizer(num_words= vocabulary_size)
    # Fitting text on the tokenizer
    tokenizer.fit_on_texts(df['text'])
    # Converting text to sequence
    sequences = tokenizer.texts_to_sequences(df['text'])
    # Finding unique tokenizer
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    # Padding sequences to the length of MAX_SEQUENCE_LENGTH
    data = pad_sequences(sequences, maxlen=50)
    labels = df['y']
    return data,labels

def split_train_test_data(data,labels):
    # splits the dataset into training and test dataset
    """
    data,labels: return output from tokenize_and_convert function
    return: X_train, X_test, y_train, y_test
    """
    # Getting the labels and features data
    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    # Getting the splitting index for training and testing data
    SPLIT_RATIO = 0.20
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    validation_samples = int(SPLIT_RATIO * data.shape[0])
    # Getting the testing and training dataset
    X_train = data[:-validation_samples]
    y_train = labels[:-validation_samples]
    X_test = data[-validation_samples:]
    y_test = labels[-validation_samples:]
    return X_train, X_test, y_train, y_test

def load_train_test_data():
    # Final function that calls all the above functions and returns the training and test dataset
    """
    return : X_train, X_test, y_train, y_test
    """
    df = read_data()
    df['text'] = df['text'].map(lambda x: clean_text(x))
    data,labels = tokenize_and_convert(df)
    X_train, X_test, y_train, y_test = split_train_test_data(data,labels)
    return X_train, X_test, y_train, y_test

################################################################################
