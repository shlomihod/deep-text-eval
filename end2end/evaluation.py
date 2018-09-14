"""
Date: 07/09/2018
Version: 2.1
Description: 1) Takes all the models and evaluates them for accuracy and threshold score.
             2) Saves them in a pandas dataframe
Python Version: 3.6
Author: Team-1b-Deep Text Eval(Shlomi Hod, Maximilian Seidler,Vageesh Saxena)
"""

########################### Importing libraries ################################
import numpy as np
import pandas as pd

from clean_and_split_data import load_train_test_data
from utils import threshold_score
from cnn import cnn_model
from han import han_model

from keras.models import Model,load_model
################################################################################

# Used to modify elements of a dictionary
class make_dict(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

def lstm(X_train, X_test, y_train, y_test):
    # Evaluates basic lstm
    """
    return : 1) accuracy on categorical_crossentropy loss function
             2) threshold_score
    """
    model = load_model("models/lstm.h5")
    # Checking the accuracy
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    # Making predictions
    prediction = model.predict_classes(X_test,batch_size=10,verbose=0)
    # Calculating accuracy and threshold score
    threshold = threshold_score(y_test.argmax(axis=-1),prediction)
    return train_accuracy[1]*100,test_accuracy[1]*100,threshold*100


def bidirectional_lstm(X_train, X_test, y_train, y_test):
    # Evaluates bidirectional_lstm model
    """
    return : 1) accuracy on categorical_crossentropy loss function
             2) threshold_score
    """
    model = load_model("models/bidirectional_lstm.h5")
    # Checking the accuracy
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    # Making predictions
    prediction = model.predict_classes(X_test,batch_size=10,verbose=0)
    # Calculating accuracy and threshold score
    threshold = threshold_score(y_test.argmax(axis=-1),prediction)
    return train_accuracy[1]*100,test_accuracy[1]*100,threshold*100

def cnn_lstm(X_train, X_test, y_train, y_test):
    # Evaluates cnn_lstm model
    """
    return : 1) accuracy on categorical_crossentropy loss function
             2) threshold_score
    """
    model = load_model("models/lstm_cnn.h5")
    # Checking the accuracy
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    # Making predictions
    prediction = model.predict_classes(X_test,batch_size=10,verbose=0)
    # Calculating accuracy and threshold score
    threshold = threshold_score(y_test.argmax(axis=-1),prediction)
    return train_accuracy[1]*100,test_accuracy[1]*100,threshold*100

def cnn_multi_filter(X_train, X_test, y_train, y_test):
    # Evaluates cnn_multi_filter model
    """
    return : 1) accuracy on categorical_crossentropy loss function
             2) threshold_score
    """
    model = load_model("models/cnn_multi-filter.h5")
    # Checking the accuracy
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    # Making predictions
    prediction = model.predict_classes(X_test,batch_size=10,verbose=0)
    # Calculating accuracy and threshold score
    threshold = threshold_score(y_test.argmax(axis=-1),prediction)
    return train_accuracy[1]*100,test_accuracy[1]*100,threshold*100

def simple_nn(X_train, X_test, y_train, y_test):
    # Evaluates simple_nn model
    """
    return : 1) accuracy on categorical_crossentropy loss function
             2) threshold_score
    """
    model = load_model("models/simple_nn.h5")
    # Checking the accuracy
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    # Making predictions
    prediction = model.predict_classes(X_test,batch_size=10,verbose=0)
    # Calculating accuracy and threshold score
    threshold = threshold_score(y_test.argmax(axis=-1),prediction)
    return train_accuracy[1]*100,test_accuracy[1]*100,threshold*100


if __name__ == '__main__':
    # Loading data
    X_train, X_test, y_train, y_test = load_train_test_data()
    # initalization of list for assigning values
    train_accuracy,test_accuracy,threshold = ([] for i in range(3))
    models = ["simple_nn","CNN","CNN multi-filter","LSTM","Bidirectional LSTM","CNN-LSTM","HAN"]
    df = pd.DataFrame(columns=["NN models","Training Accuracy","Testing accuracy","threshold_score"])
    train_acc,test_acc,thresh = simple_nn(X_train, X_test, y_train, y_test)
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)
    threshold.append(thresh)
    train_acc,test_acc,thresh = cnn_model()
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)
    threshold.append(thresh)
    train_acc,test_acc,thresh = cnn_multi_filter(X_train, X_test, y_train, y_test)
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)
    threshold.append(thresh)
    train_acc,test_acc,thresh = lstm(X_train, X_test, y_train, y_test)
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)
    threshold.append(thresh)
    train_acc,test_acc,thresh = bidirectional_lstm(X_train, X_test, y_train, y_test)
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)
    threshold.append(thresh)
    train_acc,test_acc,thresh = cnn_lstm(X_train, X_test, y_train, y_test)
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)
    threshold.append(thresh)
    train_acc,test_acc,thresh = han_model()
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)
    threshold.append(thresh)
    df["NN models"] = models
    df["Training Accuracy"] = train_accuracy
    df["threshold_score"] = threshold
    df["Testing accuracy"] = test_accuracy
    df.to_hdf('results.h5', "evaluated_results", table=True, mode='a')

######################################################################################
