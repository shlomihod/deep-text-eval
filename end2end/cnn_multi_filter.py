"""
Date: 04/09/2018
Version: 2.1
Description: Text Classification using Using Convolutional neural network with multiple filter sizes
             (http://www.aclweb.org/anthology/D14-1181)
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

from keras.layers import Conv1D, Flatten, MaxPooling1D, Dropout, Activation, Input, Dense, concatenate
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
    filter_sizes = (2,4,5,8)
    dropout_prob = [0.4,0.5]
    # Setting the Convolution layer
    graph_in = Input(shape=(max_input_length, embedding_dim))
    convs = []
    avgs = []

    for fsz in filter_sizes:
        conv = Conv1D(nb_filter=32,filter_length=fsz,border_mode='valid',activation='relu',subsample_length=1)(graph_in)
        pool = MaxPooling1D(pool_length = max_input_length - fsz + 1)(conv)
        flattenMax = Flatten()(pool)
        convs.append(flattenMax)

    if len(filter_sizes)>1:
        out = concatenate(convs,axis=-1)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out, name="graphModel")
    # Un-comment the below mentioned code to train your model

    # Configuring the neural network
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_size, output_dim = embedding_dim,input_length = max_input_length,trainable=True))
    model.add(Dropout(dropout_prob[0]))
    model.add(graph)
    model.add(Dense(256))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(y_train.shape[1]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    print("model fitting - CNN network")
    model.summary()
    # Training the model
    history = model.fit(X_train,y_train,validation_data=(X_test, y_test),epochs=10)
    plot_accuracy_and_loss_curves(history,"categorical_crossentropy","cnn_multiple_filter")
    # Saving the model
    model.save("models/cnn_multi-filter.h5")
    plot_model(model, to_file='model_images/cnn_multi-filter_model.png', show_shapes=True, show_layer_names=True)

################################################################################
