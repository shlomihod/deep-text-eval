"""
Date: 13th August, 2018
Time: 12:23
Version: 1.1
Description: Returns the scores using Textstat function
python version: 3.6
Dataset: Wikipedia Corpus
Preface :
A) Syllable Count : Returns the number of syllables present in the given text.
B) Lexicon Count : Calculates the number of words present in the text.
C) Flesch Reading Ease score : Returns the Flesch Reading Ease Score.
                               While the maximum score is 121.22, there is no
                               limit on how low the score can be. A negative
                               score is valid.
D) Flesch-Kincaid Grade Level : Returns the Flesch-Kincaid Grade of the given text.
                                This is a grade formula in that a score of 9.3 means
                                that a ninth grader would be able to read the document.
E)Fog Scale : Returns the FOG index of the given text. This is a grade formula
              in that a score of 9.3 means that a ninth grader would be able to
              read the document.
F)SMOG Index : Returns the SMOG index of the given text. This is a grade formula
               in that a score of 9.3 means that a ninth grader would be able to
               read the document.
G)Automated Readability Index : Returns the ARI (Automated Readability Index)
                                which outputs a number that approximates the grade
                                level needed to comprehend the text.
H)Coleman-Liau Index : Returns the grade level of the text using the
                       Coleman-Liau Formula. This is a grade formula in that a
                       score of 9.3 means that a ninth grader would be able to
                       read the document.
I)Linsear Write Formula : Returns the grade level using the Linsear Write Formula.
                          This is a grade formula in that a score of 9.3 means that
                          a ninth grader would be able to read the document.
J)Dale-Chall Readability Score : Different from other tests, since it uses a lookup
                                 table of the most commonly used 3000 English words.
                                 Thus it returns the grade level using the New
                                 Dale-Chall Formula.
K)Readability Consensus : Based upon all the above tests, returns the estimated
                          school grade level required to understand the text.
L) Difficult words : calculates the number of difficult words in a given sentence.
"""

##################### Installing the dependencies ##############################
# ! git clone https://github.com/shivam5992/textstat.git
# ! cd textstat
# ! pip install .
# ! pip install pandas
# ! pip install numpy
################################################################################


#################### Importing the libraries ###################################
import os
from collections import defaultdict

import textstat
from textstat.textstat import textstatistics, easy_word_set, legacy_round
import numpy as np
import pandas as pd
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

# Returns Number of Words in the text
def word_count(text):
    words = 0
    for sentence in text:
        words += len([token for token in sentence])
    return words

# Returns the number of sentences in the text
def sentence_count(text):
    return len(text)

# Returns average sentence length
def avg_sentence_length(text):
    words = word_count(text)
    sentences = sentence_count(text)
    average_sentence_length = float(words / sentences)
    return average_sentence_length

# Textstat is a python package, to calculate statistics from
# text to determine readability,
# complexity and grade level of a particular corpus.
# Package can be found at https://pypi.python.org/pypi/textstat
def syllables_count(word):
    return textstatistics().syllable_count(word,lang='en_US')

# Returns the average number of syllables per
# word in the text
def avg_syllables_per_word(text):
    syllable = syllables_count(text)
    words = word_count(text)
    ASPW = float(syllable) / float(words)
    return legacy_round(ASPW, 1)

# Return total Difficult Words in a text
def difficult_words(text):
    # Find all words in the text
    words = []
    for sentence in text:
        words += [token for token in sentence]

    # difficult words are those with syllables >= 2
    # easy_word_set is provide by Textstat as
    # a list of common words
    diff_words_set = set()

    for word in words:
        syllable_count = syllables_count(word)
        if word not in easy_word_set and syllable_count >= 2:
            diff_words_set.add(word)

    return len(diff_words_set)

# A word is polysyllablic if it has more than 3 syllables
# this functions returns the number of all such words
# present in the text
def poly_syllable_count(text):
    count = 0
    words = []
    for sentence in text:
        words += [token for token in sentence]


    for word in words:
        syllable_count = syllables_count(word)
        if syllable_count >= 3:
            count = 1
    return count


def flesch_reading_ease(text):
    """
        Implements Flesch Formula:
        Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW)
        Here,
          ASL = average sentence length (number of words
                divided by number of sentences)
          ASW = average word length in syllables (number of syllables
                divided by number of words)
    """
    FRE = 206.835 - float(1.015 * avg_sentence_length(text)) -\
          float(84.6 * avg_syllables_per_word(text))
    return legacy_round(FRE, 2)


def gunning_fog(text):
    per_diff_words = (difficult_words(text) / word_count(text) * 100) + 5
    grade = 0.4 * (avg_sentence_length(text) + per_diff_words)
    return grade


def smog_index(text):
    """
        Implements SMOG Formula / Grading
        SMOG grading = 3 + ?polysyllable count.
        Here,
           polysyllable count = number of words of more
          than two syllables in a sample of 30 sentences.
    """

    if sentence_count(text) >= 3:
        poly_syllab = poly_syllable_count(text)
        SMOG = (1.043 * (30*(poly_syllab / sentence_count(text)))**0.5) \
                + 3.1291
        return legacy_round(SMOG, 1)
    else:
        return 0


def dale_chall_readability_score(text):
    """
        Implements Dale Challe Formula:
        Raw score = 0.1579*(PDW) + 0.0496*(ASL) + 3.6365
        Here,
            PDW = Percentage of difficult words.
            ASL = Average sentence length
    """
    words = word_count(text)
    # Number of words not termed as difficult words
    count = word_count(text) - difficult_words(text)
    if count > 0:

        # Percentage of words not on difficult word list

        per = float(count) / float(words) * 100

    # diff_words stores percentage of difficult words
    diff_words = 100 - per

    raw_score = (0.1579 * diff_words) + \
                (0.0496 * avg_sentence_length(text))

    # If Percentage of Difficult Words is greater than 5 %, then;
    # Adjusted Score = Raw Score + 3.6365,
    # otherwise Adjusted Score = Raw Score

    if diff_words > 5:

        raw_score += 3.6365

    return legacy_round(raw_score, 2)

# returns total number of lexical_counts in a given sentence
def lexical_counts(sent):
    return textstat.lexicon_count(sent, removepunct=True)

if __name__ == '__main__':
    current_directory = os.getcwd()
    sentences = read_files(current_directory + "/data/valid_src")
    """col_names = [
    "Sentences","Word count","Average Sentence length","Syllable Count",
    "Average syllables per words","Polysyllablic count","Lexicon Count","Difficult Words",
    "Flesch Reading Ease score","Flesch-Kincaid Grade Level","Fog Scale","SMOG Index",
    "Automated Readability Index","Coleman-Liau Index","Linsear Write Formula",
    "Dale-Chall Readability Score","Readability Consensus"]"""
    df = pd.DataFrame(columns=col_names)
    df["Sentences"] = sentences
    df["Word count"] = df["Sentences"].apply(lambda x:word_count(x))
    df["Sentence Length"] = df["Sentences"].apply(lambda x:sentence_count(x))
    df["Average Sentence length"] = df["Sentences"].apply(lambda x:avg_sentence_length(x))
    df["Syllable Count"] = df["Sentences"].apply(lambda x: syllables_count(x))
    df["Average syllables per words"] = df["Sentences"].apply(lambda x: avg_syllables_per_word(x))
    df["Polysyllablic count"] = df["Sentences"].apply(lambda x:poly_syllable_count(x))
    df["Lexicon Count"] = df["Sentences"].apply(lambda x:lexical_counts(x))
    df["Flesch Reading Ease score"] = df["Sentences"].apply(lambda x:flesch_reading_ease(x))
    df["Flesch-Kincaid Grade Level"] = df["Sentences"].apply(lambda x:textstat.flesch_kincaid_grade(x))
    df["Fog Scale"] = df["Sentences"].apply(lambda x:gunning_fog(x))
    df["SMOG Index"] = df["Sentences"].apply(lambda x:smog_index(x))
    df["Automated Readability Index"] = df["Sentences"].apply(lambda x:textstat.automated_readability_index(x))
    df["Coleman-Liau Index"] = df["Sentences"].apply(lambda x:textstat.coleman_liau_index(x))
    df["Linsear Write Formula"] = df["Sentences"].apply(lambda x:textstat.linsear_write_formula(x))
    df["Dale-Chall Readability Score"] = df["Sentences"].apply(lambda x:dale_chall_readability_score(x))
    df["Readability Consensus"] = df["Sentences"].apply(lambda x:textstat.text_standard(x,float_output=False))
    df.to_hdf('textstat_data.h5', key='textstat', mode='w')
