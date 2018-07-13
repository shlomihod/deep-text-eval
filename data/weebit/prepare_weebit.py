import glob
import itertools as it
from collections import Counter, defaultdict
from pprint import pprint

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from langdetect import detect_langs
from sklearn.model_selection import train_test_split

RANDOM_STATE = 70988860

N_BitGCSE = 800

TEST_SIZE = 0.2

FIELDS = ['path', 'source', 'number']

LEVELS = {'WRLevel2': 0, # Age 7–8
          'WRLevel3': 1, # Age 8-9
          'WRLevel4': 2, # Age 9-10
          'BitKS3': 3,   # Age 11-14
          'BitGCSE': 4}  # Age 14-16

WR_NON_CONTENT_LINES = ['All trademarks and logos are property of Weekly Reader Corporation.',
                'measures published under license with MetaMetrics, Inc.']

BBC_NON_CONTENT_LINES = ['This page is best viewed in an up-to-date web browser with style sheets (CSS) enabled.',
                        'While you will be able to view the content of this page in your current browser, you will not be able to get the full visual experience.',
                         'Please consider upgrading your browser software or enabling style sheets (CSS) if you are able to do so.',
                         'The BBC is not responsible for the content of external internet sites.',
                         'For information on how to enable JavaScript please go to th.',
                         'You will not be able to see this content until you have JavaScript switched on.',
                         'Your web browser does not have JavaScript switched on at the moment.',
                         'You have disabled Javascript, or are not running Javascript on this browser.',
                         'Go to th.',
                         'go to th.',
                         'The enhanced version of the site requires the Flash 8 plugin (or higher) to be installed and JavaScript to be enabled on your browser.',
                         'To find out how to turn on JavaScript',
                         'The enhanced version of the site requires the Flash 8 plugin (or higher) to be installed and JavaScript to be enabled on your browser.',
                         'To find out how to install a Flash plugin,',
                         'The enhanced version of the site requires the Flash 8 plugin (or higher) to be installed and JavaScript to be enabled on your browser.',
                         'Download the Adobe Flash player to view this conten.',
                        ]


NON_CONTENT_LINES = WR_NON_CONTENT_LINES + BBC_NON_CONTENT_LINES

nlp = spacy.util.get_lang_class('en')()
nlp.add_pipe(nlp.create_pipe('sentencizer'))

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


def filter_start_dups(text_df):
    """Removing all the texts that share a start with another text, with more than 4 lines from the beginning.
    This influences almost only on our biggest class by far, so it doesn't make our corpus smaller (BitGCSE)"""
    
    docs = [doc for doc in nlp.pipe(tqdm(text_df['text']))]

    all_lines = sum([list([sent.string.strip() for sent in doc.sents]) for doc in docs], [])
    lines_count = Counter(all_lines)
    just_lines = [list([str(sent) for sent in doc.sents]) for doc in docs]

    print('=== Most Common Lines ===')
    pprint(lines_count.most_common(15))
    print('=== End ===\n')
    
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

def downsample_BitGCSE(text_df):

    text_df_BitGCSE = text_df[text_df['source'] == 'BitGCSE']
    text_df_NOT_BitGCSE = text_df[text_df['source'] != 'BitGCSE']

    text_df_BitGCSE = text_df_BitGCSE.sample(N_BitGCSE,
                                             replace=False,
                                             random_state=RANDOM_STATE)

    text_df = pd.concat([text_df_BitGCSE, text_df_NOT_BitGCSE])
    
    print('=== Source Counts ===')
    print(text_df['source'].value_counts())
    print('=== End ===\n')
                
    return text_df


def print_data_counts(text_df, train_df, test_df):
    text_counts = text_df['y'].value_counts()
    train_counts = train_df['y'].value_counts()
    test_counts = test_df['y'].value_counts()

    print(pd.DataFrame({
        '#text': text_counts,
        '%text': (100 * text_counts / text_counts.sum()).round(2),
        '#train': train_counts,
        '%train': (100 * train_counts / train_counts.sum()).round(2),
        '#test': test_counts,
        '%test': (100 * test_counts / test_counts.sum()).round(2),
    }).sort_index())
    

def prepare_corpus():
    text_df = pd.DataFrame((dict(zip(FIELDS,
                    [article] + article.split('/')[1:]))
                 for article in glob.glob('WeeBit-TextOnly/*/*')))

    text_df['number'] = text_df['number'].str[:-4]
    
    text_df['y'] = text_df['source'].map(LEVELS)

    print('Reading files...')
    text_df['text'] = (text_df['path'].apply(read_content).str
                                                          .replace('.\n', '. ')
                                                          .replace('\n', '. ')
                                                          .apply(remove_non_content_lines))
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

    print('Downsample BitGCSE...')
    text_df = downsample_BitGCSE(text_df)
    
    print('Reset Index...')
    text_df = text_df.reset_index(drop=True)

    print('Train-Test Split...')
    train_df, test_df = train_test_split(text_df,
                                     test_size=TEST_SIZE,
                                     shuffle=True,
                                     stratify=text_df['y'],
                                     random_state=RANDOM_STATE)
    
    print_data_counts(text_df, train_df, test_df)
    
    return text_df, train_df, test_df


def main():
    text_df, train_df, test_df = prepare_corpus()
    
    with pd.HDFStore('weebit.h5', mode='w') as store:
        store.put('text_df', text_df)
        store.put('train_df', train_df)
        store.put('test_df', test_df)


if __name__ == '__main__':
    main()