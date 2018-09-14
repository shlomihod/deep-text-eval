"""
Date: 04/09/2018
Version: 2.1
Description: Text Classification using Using a bidirectional LSTM based neural network
             with glove word embeddings.
             (http://aclweb.org/anthology/C18-1180)
Glove Data: https://nlp.stanford.edu/projects/glove/
Python Version: 3.6
Author: Team-1b-Deep Text Eval(Shlomi Hod, Maximilian Seidler,Vageesh Saxena)
"""

####################### Importing libraries ####################################
import os
import numpy as np

from clean_and_split_data import load_train_test_data
from utils import plot_accuracy_and_loss_curves

from keras.layers import LSTM, Dense, Input, Dropout, Bidirectional
from keras.models import Model,load_model
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.utils.vis_utils import plot_model
################################################################################

if __name__ == '__main__':
    # Loading data
    X_train, X_test, y_train, y_test = load_train_test_data()
    # defining the hyper-parameters
    max_input_length = 50
    vocabulary_size = 20000
    embedding_dim = 100
    model = Sequential()
    model.add(Embedding(vocabulary_size, 100, input_length=max_input_length))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    # try using different optimizers and different optimizer configs
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    print("model fitting - LSTM with dropout")
    model.summary()
    history = model.fit(X_train, y_train,epochs=10,validation_data=[X_test, y_test])
    plot_accuracy_and_loss_curves(history,"categorical_crossentropy","bidirectional_lstm")
    # Saving the model
    model.save("models/bidirectional_lstm.h5")
    plot_model(model, to_file='model_images/bidirectional_lstm_model.png', show_shapes=True, show_layer_names=True)

################################################################################
