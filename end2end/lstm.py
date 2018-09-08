"""
Date: 04/09/2018
Version: 1.1
Description: Text Classification using Using a LSTM based neural network.
             (https://arxiv.org/abs/1607.02501)
Python Version: 3.6
Author: Team-1b-Deep Text Eval(Shlomi Hod, Maximilian Seidler,Vageesh Saxena)
"""

####################### Importing libraries ####################################
from clean_and_split_data import load_train_test_data
from utils import plot_accuracy_and_loss_curves

from keras.layers import LSTM, Dense
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
    # Configuring the neural network
    model  = Sequential()
    model.add(Embedding(vocabulary_size, 100, input_length=max_input_length))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print("model fitting - LSTM network")
    model.summary()
    # Training the model
    history = model.fit(X_train,y_train,validation_data=(X_test, y_test),epochs=10)
    plot_accuracy_and_loss_curves(history,"categorical_crossentropy","lstm")
    model.save("models/lstm.h5")
    plot_model(model, to_file='model_images/lstm_model.png', show_shapes=True, show_layer_names=True)
################################################################################
