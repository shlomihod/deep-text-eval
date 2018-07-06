import glob

import numpy as np
import pandas as pd


FIELDS = ['path', 'source', 'number']

LEVELS = {'WRLevel2': 0, # Age 7–8
          'WRLevel3': 1, # Age 8-9
          'WRLevel4': 2, # Age 9-10
          'BitKS3': np.nan,   # Age 11-14
          'BitGCSE': np.nan}  # Age 14-16

NON_CONTENT_LINES = ['All trademarks and logos are property of Weekly Reader Corporation.',
                'measures published under license with MetaMetrics, Inc.']


def read_content(path):
    with open(path, 'r', encoding='latin-1') as f:
            return f.read()

        
def remove_non_content_lines(text):
    for line in NON_CONTENT_LINES:
        text = text.replace(line, '')
    return text.strip()


def main():
    text_df = pd.DataFrame((dict(zip(FIELDS,
                    [article] + article.split('/')[1:]))
                 for article in glob.glob('WeeBit-TextOnly/*/*')))

    text_df['number'] = text_df['number'].str[:-4]
    
    text_df['y'] = text_df['source'].map(LEVELS)
    text_df = text_df.dropna()
    text_df['text'] = text_df['path'].apply(read_content).apply(remove_non_content_lines)
    text_df = text_df[text_df['text'].str.len() != 0]
    
    text_df.to_hdf("weebit.h5", "text_df")

if __name__ == "__main__":
    main()