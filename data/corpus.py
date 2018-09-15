__all__ = ['load_corpus']

import os

import pandas as pd
from sklearn.preprocessing import scale
from keras.utils import to_categorical


def load_corpus(corpus_name):
    
    path = os.path.join(os.path.dirname(__file__), corpus_name, corpus_name + '.h5')

    text_features_df = pd.read_hdf(path, 'text_features_df')
    train_features_df = pd.read_hdf(path, 'train_features_df')
    test_features_df = pd.read_hdf(path, 'test_features_df')

    features_mask = text_features_df.columns.str.startswith('feature_')
    y_mask = text_features_df.columns == 'y'
    features_y_mask = features_mask | y_mask

    feature_names = text_features_df.columns[features_mask]

    X_all = text_features_df.loc[:, features_mask]
    y_all = text_features_df['y']
    y_all_onehot = to_categorical(y_all)

    X_train = train_features_df.loc[:, features_mask]
    y_train = train_features_df['y']
    y_train_onehot = to_categorical(y_train)

    X_test = test_features_df.loc[:, features_mask]
    y_test = test_features_df['y']
    y_test_onehot = to_categorical(y_test)

    X_all = scale(X_all)

    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)

    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    
    FEATURES_NAMES = {'POS_DENSITY': ['feature_' + name for name in ['num_comma', 'nouns', 'propernouns', 'pronouns', 'conj', 'adj', 'ver', 'interj', 'adverbs', 'modals', 'perpro' , 'whpro', 'numfuncwords', 'numdet', 'numvb' , 'numvbd', 'numvbg', 'numvbn', 'numvbp', 'numprep']],
                  'SYNTACTIC_COMPLEXITY': ['feature_' + name for name in ['senlen', 'numnp', 'numpp', 'numvp', 'numsbar', 'numsbarq', 'numwh', 'avgnpsize', 'avgvpsize', 'avgppsize', 'avgparsetreeheight', 'numconstituents']],
                  'READABILITY_SCORES': [name for name in X_train.columns if name.startswith('feature_rs_')],
                 }


    return {'text_features_df': text_features_df,
            'features_mask': features_mask,
            'features_y_mask': features_y_mask,
            'feature_names': feature_names,
            
            'X_all': X_all,
            'y_all': y_all,
            'y_all_onehot': y_all_onehot,
            
            'X_train': X_train,
            'y_train': y_train,
            'y_train_onehot': y_train_onehot,
            
            'X_test': X_test,
            'y_test': y_test,
            'y_test_onehot': y_test_onehot,
            
            'FEATURES_NAMES': FEATURES_NAMES,
           }