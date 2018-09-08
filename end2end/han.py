"""
Date: 06/09/2018
Version: 1.1
Description: Text Classification using Using a Hierarchical Attention Networks.
             (https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
Python Version: 3.6
Author: Team-1b-Deep Text Eval(Shlomi Hod, Maximilian Seidler,Vageesh Saxena)
"""

####################### Importing libraries ####################################
import re
import logging
import sys

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model

from utils import HAN,AttentionLayer,plot_accuracy_and_loss_curves,threshold_score
################################################################################

def han_model():
    """
    return : threshold_score,accuracy
    """
    # Create a logger to provide info on the state of the
    # script
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger = logging.getLogger('default')
    logger.setLevel(logging.INFO)
    logger.addHandler(stdout)

    MAX_WORDS_PER_SENT = 100
    MAX_SENT = 15
    MAX_VOC_SIZE = 20000
    GLOVE_DIM = 100
    TEST_SPLIT = 0.2


    ################################################################################
    # Pre processing
    ################################################################################
    logger.info("Pre-processsing data.")

    # Load Kaggle's IMDB example data
    data = pd.read_hdf("weebit.h5","text_df")[['text','y']]
    # Dropping null values
    data.dropna(inplace=True)
    # Converting class labels to int dtype
    data['y'] = data['y'].astype(int)
    print("Shape of the dataset:",data.shape)


    # Do some basic cleaning of the text text
    def remove_quotations(text):
        """
        Remove quotations and slashes from the dataset.
        """
        text = re.sub(r"\\", "", text)
        text = re.sub(r"\'", "", text)
        text = re.sub(r"\"", "", text)
        return text


    def remove_html(text):
        """
        Very, very raw parser to remove HTML tags from
        texts.
        """
        tags_regex = re.compile(r'<.*?>')
        return tags_regex.sub('', text)


    data['text'] = data['text'].apply(remove_quotations)
    data['text'] = data['text'].apply(remove_html)
    data['text'] = data['text'].apply(lambda x: x.strip().lower())

    # Get the data and the y
    texts = data['text'].values
    target = data['y'].values
    del data


    ################################################################################
    # Tokenization
    ################################################################################
    logger.info("Tokenization.")

    # Build a Keras Tokenizer that can encode every token
    word_tokenizer = Tokenizer(num_words=MAX_VOC_SIZE)
    word_tokenizer.fit_on_texts(texts)

    # Construct the input matrix. This should be a nd-array of
    # shape (n_samples, MAX_SENT, MAX_WORDS_PER_SENT).
    # We zero-pad this matrix (this does not influence
    # any predictions due to the attention mechanism.
    X = np.zeros((len(texts), MAX_SENT, MAX_WORDS_PER_SENT), dtype='int32')

    for i, text in enumerate(texts):
        sentences = sent_tokenize(text)
        tokenized_sentences = word_tokenizer.texts_to_sequences(
            sentences
        )
        tokenized_sentences = pad_sequences(
            tokenized_sentences, maxlen=MAX_WORDS_PER_SENT
        )

        pad_size = MAX_SENT - tokenized_sentences.shape[0]

        if pad_size < 0:
            tokenized_sentences = tokenized_sentences[0:MAX_SENT]
        else:
            tokenized_sentences = np.pad(
                tokenized_sentences, ((0,pad_size),(0,0)),
                mode='constant', constant_values=0
            )

        # Store this observation as the i-th observation in
        # the data matrix
        X[i] = tokenized_sentences[None, ...]

    # Transform the labels into a format Keras can handle
    y = to_categorical(target)

    # We make a train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT)


    ################################################################################
    # Word Embeddings
    ################################################################################
    logger.info(
        "Creating embedding matrix using pre-trained GloVe vectors."
    )

    # Now, we need to build the embedding matrix. For this we use
    # a pretrained (on the wikipedia corpus) 100-dimensional GloVe
    # model.

    # Load the embeddings from a file
    embeddings = {}
    with open('glove.6B/glove.6B.100d.txt', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')

            embeddings[word] = coefs

    # Initialize a matrix to hold the word embeddings
    embedding_matrix = np.random.random(
        (len(word_tokenizer.word_index) + 1, GLOVE_DIM)
    )

    # Let the padded indices map to zero-vectors. This will
    # prevent the padding from influencing the results
    embedding_matrix[0] = 0

    # Loop though all the words in the word_index and where possible
    # replace the random initalization with the GloVe vector.
    for word, index in word_tokenizer.word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector


    ################################################################################
    # Model Training
    ################################################################################
    logger.info("Training the model.")


    han_model = HAN(
        MAX_WORDS_PER_SENT, MAX_SENT, 5, embedding_matrix,
        word_encoding_dim=100, sentence_encoding_dim=100
    )

    han_model.summary()

    han_model.compile(
        optimizer='adam', loss='categorical_crossentropy',
        metrics=['acc']
    )

    history = han_model.fit(
        X_train, y_train, epochs=10,
        validation_data=(X_test, y_test)
    )

    plot_accuracy_and_loss_curves(history,"categorical_crossentropy","han")
    han_model.save("models/han.h5")
    plot_model(han_model, to_file='model_images/han_model.png', show_shapes=True, show_layer_names=True)
    accuracy = han_model.evaluate(X_test, y_test, verbose=0)
    # Calculating accuracy and threshold score
    # Making predictions
    prediction = han_model.predict(X_test,batch_size=10,verbose=0)
    prediction = prediction.argmax(axis=-1)
    threshold = threshold_score(y_test.argmax(axis=-1),prediction)
    return accuracy[1]*100,threshold*100
################################################################################
