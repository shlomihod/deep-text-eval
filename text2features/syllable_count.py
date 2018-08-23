"""
Date: 13th August, 2018
Time: 17:03
Version: 1.1
Description: Returns the scores using Textstat function
python version: 3.6
Dataset: Wikipedia Corpus
"""

##################### Installing the dependencies ##############################
# ! git clone https://github.com/shivam5992/textstat.git
# ! cd textstat
# ! pip install .
################################################################################


####################### Importing the libraries ################################
import os
from collections import defaultdict

import textstat
from textstat.textstat import textstatistics
################################################################################

def read_files(file_name):
    # read all the files from a given directory
    """
    file_name : takes a file_name address along with the extension(dytpe=string)
    return : list of sentences in the file(dtype = list).
    """
    #For removing whitespace characters like `\n` at the end of each line
    lines = [line.rstrip('\n') for line in open(file_name)]
    return lines

# Textstat is a python package, to calculate statistics from
# text to determine readability,
# complexity and grade level of a particular corpus.
# Package can be found at https://pypi.python.org/pypi/textstat
def syllables_count(word):
    return (word,textstatistics().syllable_count(word,lang='en_US'))

# Used to modify elements of a dictionary
class make_dict(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

if __name__ == '__main__':
    syl_dict = make_dict()
    current_directory = os.getcwd()
    words = read_files(current_directory + "/data/src-vocab.txt")
    for word in words:
        temp = syllables_count(word)
        syl_dict[temp[0]] = temp[1]

    print(syl_dict)
