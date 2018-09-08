"""
Date: 03/09/2018
Version: 1.1
Description: Provides a visualization of the corpus data
Python Version: 3.6
Author: Team-1b-Deep Text Eval(Shlomi Hod, Maximilian Seidler,Vageesh Saxena)
"""

####################### Importing libraries ####################################
from clean_and_split_data import read_data,clean_text

import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib import pyplot
################################################################################

# Used to modify elements of a dictionary
class make_dict(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

if __name__ == '__main__':
    df = read_data()
    df['text'] = df['text'].map(lambda x: clean_text(x))
    # Getting all the unique class labels
    unique_class_labels = df['y'].unique()
    print("class labels from the corpora:",unique_class_labels)
    # Printing the total number of sentences in the corpora
    print("Total no of sentences in the Corpora is",df.shape[0])
    # Getting count of sentences for every class
    count_list = []
    for class_label in df['y'].unique():
        print("No of sentences in class label",str(class_label) + " is " + str(df[df['y']==class_label].shape[0]))
        count_list.append((str(class_label),int(df[df['y']==class_label].shape[0])))
    # Getting data into dictionary for plotting
    no_of_sentence = make_dict()
    for values in count_list:
        no_of_sentence[values[0]] = values[1]
    # plotting the data
    plt.bar(list(no_of_sentence.keys()), no_of_sentence.values(), color='g')
    plt.suptitle(no_of_sentence.keys(), fontsize=10)
    plt.xlabel('Class labels', fontsize=10)
    plt.ylabel('No of Sentences', fontsize=10)
    plt.savefig('images/visualize-pretrained_data')
    plt.show()

################################################################################
