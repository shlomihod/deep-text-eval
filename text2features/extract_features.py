"""
Date: 09 September, 2018
Version: 1.1
Description: Extracts the features for given wiki sentences as per the Brown Corpus
python version: 3.6
Dataset: Brown corpus
         Wiki corpus
Author: Team-1b-Deep Text Eval(Shlomi Hod, Maximilian Seidler,Vageesh Saxena)
POS tags used :
CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: “there is” … think of it like “there exists”)
FW foreign word
IN preposition/subordinating conjunction
JJ adjective ‘big’
JJR adjective, comparative ‘bigger’
JJS adjective, superlative ‘biggest’
LS list marker 1)
MD modal could, will
NN noun, singular ‘desk’
NNS noun plural ‘desks’
NNP proper noun, singular ‘Harrison’
NNPS proper noun, plural ‘Americans’
PDT predeterminer ‘all the kids’
POS possessive ending parent’s
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO, to go ‘to’ the store.
UH interjection, errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where, when
"""

#################### Importing the libraries ###################################
import os
import nltk
nltk.download('brown')
from nltk.corpus import brown
from collections import defaultdict, Counter
import operator

import numpy as np
import pandas as pd
################################################################################

def split_wiki_corpus(corpus):
    # read all the files from a given directory
    """
    corpus : takes a corpus
    return : list of stripped sentences.
    """
    #For removing whitespace characters like `\n` at the end of each line
    lines = [line.split() for line in corpus]
    return lines

def return_POS_with_most_value(word_tags,wiki_sent):
    # returns the most prob POS
    """
    word_tags : brown corpus counter dictionary
    wiki_sent : to be tagged
    return : The most probable tags for each word in the sentence along with
             their occurence frequency
    """
    tag_prob_list = []
    for sent in wiki_sent:
        word_list = []
        for words in sent:
            # Getting all the possible POS tags for a given word
            tagged_word = word_tags[words]
            tag,count = ([] for i in range(2))
            # If there exists a tag
            dict = defaultdict()
            if tagged_word:
                # Just copy the key and values to list tag and count
                for key,values in tagged_word.items():
                    if key != "NIL":
                        tag.append(key[0:2])
                        count.append(values)
                    else:
                        pass
                # Make all sorts of Noun as Noun, all sorts of verbs as Verb and so on.
                tag = [w.replace('NP', 'NN').replace("WP","PR").replace("WD","DT") for w in tag]
                unique_tags = set(tag)
                # For each of these unique tags calculate the count
                for values in unique_tags:
                    unique_index = [index for index, value in enumerate(tag) if value == values]
                    occurence = 0
                    for indexes in unique_index:
                        occurence += count[indexes]
                    dict[values] = occurence
                word_list.append((max(dict),max(dict.values())/sum(dict.values())))
            tag_prob_list.append(word_list)
        return tag_prob_list


if __name__ == '__main__':
    # Keeps words and pos into a dictionary where the key is a word and
    # the value is a counter of POS and counts
    word_tags = defaultdict(Counter)
    for word, pos in brown.tagged_words():
        word_tags[word][pos] +=1
    # reading the wiki corpus
    wiki_corpus = list(open("data/valid_src", 'r'))
    wiki_sent = split_wiki_corpus(wiki_corpus)
    tag_prob_list = return_POS_with_most_value(word_tags,wiki_sent)
    print(tag_prob_list)
################################################################################
