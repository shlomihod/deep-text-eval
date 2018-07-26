import os

import pandas as pd
from sklearn.preprocessing import scale
from keras.utils import to_categorical

weebit_path = os.path.join(os.path.dirname(__file__), 'weebit.h5')

text_features_df = pd.read_hdf(weebit_path, 'text_features_df')
train_features_df = pd.read_hdf(weebit_path, 'train_features_df')
test_features_df = pd.read_hdf(weebit_path, 'test_features_df')

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