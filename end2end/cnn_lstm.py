"""
Date: 04/09/2018
Version: 2.1
Description: Text Classification using Using Convolutional neural network with multiple filter sizes
             (https://arxiv.org/abs/1511.08630)
Python Version: 3.6
Author: Team-1b-Deep Text Eval(Shlomi Hod, Maximilian Seidler,Vageesh Saxena)
"""

# The network starts with an embedding layer. The layer lets the system expand
# each token to a more massive vector, allowing the network to represent a word
# in a meaningful way. The layer takes 20000 as the first argument, which is the
# size of our vocabulary, and 100 as the second input parameter, which is the
# dimension of the embeddings. The third parameter is the input_length of 50,
# which is the length of each text sequence.

####################### Importing libraries ####################################
from clean_and_split_data import load_train_test_data
from utils import plot_accuracy_and_loss_curves

from keras.layers import LSTM, Conv1D, Flatten, MaxPooling1D, Dropout, Activation, Input, Dense, concatenate
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
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(100))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("model fitting - CNN-LSTM convolutional neural network")
    model.summary()
    # Training the model
    history = model.fit(X_train, y_train,validation_data=(X_test,y_test),epochs=10)
    plot_accuracy_and_loss_curves(history,"categorical_crossentropy","cnn_lstm")
    # Saving the model
    model.save("models/lstm_cnn.h5")
    plot_model(model, to_file='model_images/tm_cnn_model.png', show_shapes=True, show_layer_names=True)

################################################################################
