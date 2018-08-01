import re
import itertools
from pprint import pprint

import pandas as pd
import requests
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy import stats


HTML_TAG_RE = re.compile('<.*?>', re.DOTALL)
MARKDOWN_IMAGE_RE = re.compile('!{0,1}\[.*?\]\s?\(.*?\)', re.DOTALL)

RANDOM_STATE = 42
MIN_N_CATEGORY_DATA = 30
TEST_SIZE = 0.2

BASE_URL = 'https://newsela.com/api/v2/articleheader/'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}


class Not200Error(Exception):
    pass


def get_newsela_api(part):
    r = requests.get(BASE_URL + part, headers=HEADERS)
        
    if r.status_code != 200:
        raise Not200Error
        
    return r


def get_group_df():
    page_jsons = []

    for page_number in tqdm(itertools.count(start=1)):
    
        result = get_newsela_api('?page={}'.format(page_number)).json()

        if not result:
            break
        
        page_jsons.append(result)
    
    group_df = pd.concat(map(pd.DataFrame, page_jsons), sort=False)
    
    return group_df


def slug2articles(slug):
    r = get_newsela_api(slug)
        
    return (pd.DataFrame(
        r.json()['articles'])
      .assign(slug=slug))


def get_text_df(slugs):
    articles = []
    errors_indices = []

    for index, slug in enumerate(tqdm(slugs)):
        try:
            articles.append(slug2articles(slug))
        except Not200Error:
            errors_indices.append(index)


    text_df = pd.concat(articles)
    
    return text_df, errors_indices

def remove_too_small_categories(text_df):
    category_counts = text_df['y_cat'].value_counts()
    categories_to_keep = category_counts.index[category_counts >= MIN_N_CATEGORY_DATA]

    print('Removed Data:')
    pprint(text_df[
            ~text_df['y_cat'].isin(categories_to_keep)
        ]['y_cat'].value_counts())
    
    text_df = text_df[text_df['y_cat'].isin(categories_to_keep)]
    return text_df


def remove_html_tags(text):
    return HTML_TAG_RE.sub(' ', text)


def remove_markdown_image_tags(text):
    return MARKDOWN_IMAGE_RE.sub(' ', text)


def print_data_stats(text_df, train_df, test_df):
    dfs = [text_df, train_df, test_df]

    stats_d = {'name': ['text', 'train', 'test'],
             '#': [len(df['y']) for df in dfs],
             'min': [df['y'].min() for df in dfs],
             'max': [df['y'].max() for df in dfs],
             'mean': [df['y'].mean() for df in dfs],
             'std': [df['y'].std() for df in dfs]
            }
    stats_df = pd.DataFrame(stats_d)
    stats_df.set_index('name')

    print('Continuous y')
    pprint(stats_df)
    print()
    print('train-test Kolmogorov-Smirnov p-value', stats.ks_2samp(train_df['y'],
                                                               test_df['y']).pvalue)
    print()

    text_counts = text_df['y_cat'].value_counts()
    train_counts = train_df['y_cat'].value_counts()
    test_counts = test_df['y_cat'].value_counts()

    print('Discrete y_cat (by percentiles)')
    print(pd.DataFrame({
        '#text': text_counts,
        '%text': (100 * text_counts / text_counts.sum()).round(2),
        '#train': train_counts,
        '%train': (100 * train_counts / train_counts.sum()).round(2),
        '#test': test_counts,
        '%test': (100 * test_counts / test_counts.sum()).round(2),
    }).sort_index())
    

def prepare_corpus():

    print('Getting slugs...')
    group_df = get_group_df()
    
    print('Filterig non english slugs...')
    print(group_df['language'].value_counts())
    group_df = group_df[group_df['language'] == 'en']
    print(group_df['language'].value_counts())
 
    print('Getting Texts...')
    text_df, errors_indices = get_text_df(group_df['slug'])
    
    if errors_indices:
        print('Error Indices:', errors_indices)

        print('Retrying Getting Texts...')
        retry_text_df, retry_errors_indices = get_text_df(group_df.loc[errors_indices, 'slug'])

        print('Error Indices:', retry_errors_indices)
        
        print('Concating Texts...')
        text_df = pd.concat([text_df, retry_text_df])

    text_df = text_df.rename({'lexile_level': 'y', 'grade_level': 'y_cat'}, axis=1)
    print('#Texts =', len(text_df))
    
    print('Removing Too Small Categories...')
    text_df = remove_too_small_categories(text_df)
    print('#Texts =', len(text_df))

    print('Removing HTML Tags, Markdown Image Tags, and Extra Dashes...')
    text_df['text'] = (text_df['text']
         .apply(remove_html_tags)
         .apply(remove_markdown_image_tags).str.replace('----', ' '))

    print('Reset Index...')
    text_df = text_df.reset_index(drop=True)

    print('Train-Test Split...')
    train_df, test_df = train_test_split(text_df,
                                     test_size=TEST_SIZE,
                                     shuffle=True,
                                     stratify=text_df['y_cat'],
                                     random_state=RANDOM_STATE)
    
    print_data_stats(text_df, train_df, test_df)
    
    return text_df, train_df, test_df


def main():
    text_df, train_df, test_df = prepare_corpus()
    
    with pd.HDFStore('newsela.h5', mode='w') as store:
        store.put('text_df', text_df)
        store.put('train_df', train_df)
        store.put('test_df', test_df)

        
if __name__ == '__main__':
    main()
