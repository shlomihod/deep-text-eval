import glob
import itertools as it
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from langdetect import detect_langs


FIELDS = ['path', 'source', 'number']

LEVELS = {'WRLevel2': 0, # Age 7–8
          'WRLevel3': 1, # Age 8-9
          'WRLevel4': 2, # Age 9-10
          'BitKS3': 3,   # Age 11-14
          'BitGCSE': 4}  # Age 14-16

NON_CONTENT_LINES = ['All trademarks and logos are property of Weekly Reader Corporation.',
                'measures published under license with MetaMetrics, Inc.']

NLP = spacy.load('en_core_web_sm')


def read_content(path):
    with open(path, 'r', encoding='latin-1') as f:
            return f.read()

        
def remove_non_content_lines(text):
    for line in NON_CONTENT_LINES:
        text = text.replace(line, '')
    return text.strip()


def get_english_detect_prob(detect_langs_result):
    return {result.lang: result.prob for result in detect_langs_result}.get('en', 0.0)


def filter_non_english(text_df, english_prob_threshold=0.99):
    langs = text_df['text'].apply(detect_langs)
    english_probs = langs.apply(get_english_detect_prob)
    text_df = text_df[english_probs > english_prob_threshold]
    return text_df


def docify(text):
    doc = NLP(text)
    return doc


def filter_start_dups(text_df):
    """Removing all the texts that share a start with another text, with more than 4 lines from the beginning.
    This influences almost only on our biggest class by far, so it doesn't make our corpus smaller (BitGCSE)"""
    
    docs = []
    for _, r in tqdm(text_df.iterrows(), total=len(text_df)):
        docs.append(docify(r['text']))

    all_lines = sum([list([sent.string.strip() for sent in doc.sents]) for doc in docs], [])
    lines_count = Counter(all_lines)
    just_lines = [list([str(sent) for sent in doc.sents]) for doc in docs]

    pos2sent2ind = defaultdict(lambda : defaultdict(set))
    for ind, doc_lines in enumerate(just_lines):
        for pos, line in enumerate(doc_lines):
            pos2sent2ind[pos][line].add(ind)

    pos2ind = defaultdict(set)
    for pos in pos2sent2ind:
        for sent in pos2sent2ind[pos]:
            pos2ind[pos].update(set(it.combinations(pos2sent2ind[pos][sent], 2)))

    alreadys = set(pos2ind[1])
    highst_pos2ind = defaultdict(set)
    for pos in range(2, max(pos2ind.keys())+1):
        for ind_pair in alreadys - pos2ind[pos]:
                highst_pos2ind[pos-1].add(ind_pair)
        alreadys &= pos2ind[pos]

    start_dup_row_indices = set()
    for k in range(4, max(highst_pos2ind.keys())):
        if highst_pos2ind[k]:
            all_inds_per_pos = set.union(*[
                set(ind_pair) for ind_pair in highst_pos2ind[k]
            ])
            
            start_dup_row_indices.update(all_inds_per_pos)
            
    start_dup_row_indices_list = list(start_dup_row_indices)
            
    text_df = text_df.drop(text_df.index[start_dup_row_indices_list])

    return text_df


def prepare_corpus():
    text_df = pd.DataFrame((dict(zip(FIELDS,
                    [article] + article.split('/')[1:]))
                 for article in glob.glob('WeeBit-TextOnly/*/*')))

    text_df['number'] = text_df['number'].str[:-4]
    
    text_df['y'] = text_df['source'].map(LEVELS)

    print('Reading files...')
    text_df['text'] = text_df['path'].apply(read_content).apply(remove_non_content_lines)
    print('#Texts =', len(text_df))
    
    print('Filtering empty texts...')
    text_df = text_df[text_df['text'].str.len() != 0]
    print('#Texts =', len(text_df))
    
    print('Removing duplicates...')
    text_df = text_df.drop_duplicates('text')
    print('#Texts =', len(text_df))
    
    print('Filter non english texts...')
    text_df = filter_non_english(text_df)
    print('#Texts =', len(text_df))
    
    print('Filter texts with duplicated start...')
    text_df = filter_start_dups(text_df)
    print('#Texts =', len(text_df))

    return text_df


def main():
    text_df = prepare_corpus()
    text_df.to_hdf("weebit.h5", "text_df")


if __name__ == "__main__":
    main()