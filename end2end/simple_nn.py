"""
Date: 06/09/2018
Version: 1.1
Description: Text Classification using Simple neural network.
             (https://arxiv.org/abs/1404.2188)
Python Version: 3.6
Author: Team-1b-Deep Text Eval(Shlomi Hod, Maximilian Seidler,Vageesh Saxena)
"""

####################### Importing libraries ####################################
from clean_and_split_data import load_train_test_data
from utils import plot_accuracy_and_loss_curves,threshold_score

from keras.layers import Dropout, Dense
from keras.models import Model,load_model
from keras.models import Sequential
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
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(max_input_length,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    model.summary()
    print("Model fitting - simple NN model")
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    # Training the model
    history = model.fit(X_train, y_train,validation_data=(X_test,y_test),epochs=10)
    # Saving the model
    model.save("models/simple_nn.h5")
    plot_accuracy_and_loss_curves(history,"categorical_crossentropy","simple_nn")
    plot_model(model, to_file='model_images/simple_nn.png', show_shapes=True, show_layer_names=True)
    
###############################################################################
