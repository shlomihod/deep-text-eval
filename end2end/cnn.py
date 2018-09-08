"""
Date: 04/09/2018
Version: 1.1
Description: Text Classification using Using Convolutional neural network.
             (https://arxiv.org/abs/1404.2188)
Python Version: 3.6
Author: Team-1b-Deep Text Eval(Shlomi Hod, Maximilian Seidler,Vageesh Saxena)
"""

####################### Importing libraries ####################################
from clean_and_split_data import load_train_test_data
from utils import plot_accuracy_and_loss_curves,threshold_score
from utils import KMaxPooling
from utils import Folding

from keras.layers import Conv1D, Flatten, Dropout, Activation, Input, Dense, ZeroPadding1D
from keras.models import Model,load_model
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.utils.vis_utils import plot_model
################################################################################

def cnn_model():
    """
    return : threshold_score,accuracy
    """
    # Loading data
    X_train, X_test, y_train, y_test = load_train_test_data()
    # defining the hyper-parameters
    max_input_length = 50
    vocabulary_size = 20000
    embedding_dim = 100
    # Configuring the neural network
    model = Sequential()
    model.add(Embedding(vocabulary_size, 100, input_length=max_input_length))
    model.add(ZeroPadding1D((49,49)))
    model.add(Conv1D(64, 50, padding="same"))
    model.add(KMaxPooling(k=5, axis=1))
    model.add(Activation("relu"))
    model.add(ZeroPadding1D((24,24)))
    model.add(Conv1D(64, 25, padding="same"))
    model.add(Folding())
    model.add(KMaxPooling(k=5, axis=1))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(y_train.shape[1], activation="softmax"))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    print("model fitting - CNN network")
    model.summary()
    # Training the model
    history = model.fit(X_train,y_train,validation_data=(X_test, y_test),epochs=10)
    plot_accuracy_and_loss_curves(history,"categorical_crossentropy","cnn")
    # Saving the model
    model.save("models/cnn.h5")
    plot_model(model, to_file='model_images/cnn_model.png', show_shapes=True, show_layer_names=True)
    accuracy = model.evaluate(X_test, y_test, verbose=0)
    # Making predictions
    prediction = model.predict_classes(X_test,batch_size=10,verbose=0)
    # Calculating accuracy and threshold score
    threshold = threshold_score(y_test.argmax(axis=-1),prediction)
    return accuracy[1]*100,threshold*100
################################################################################
