import re
import glob

import numpy as np
import pandas as pd


FIELDS = ["path", "magazine", "category", "name"]

LEVELS = {"GEO": 1, "GEOlino": 0}


def read_content(path):
    with open(path, "r") as f:
        return f.read()


def preprocess_text(text):
    text = re.sub("(  \.)+", "\n", text)
    text = re.sub("\.[ ]+", ". ", text).strip()
    return text


def main():
    text_df = pd.DataFrame((dict(zip(FIELDS,
                [article] + article.split("/")[2:]))
             for article in glob.glob("./COLING_workingSet/*/*/*")))

    text_df["y"] = text_df["magazine"].map(LEVELS)

    text_df["text"] = text_df["path"].apply(read_content)
    text_df["text"] = text_df["text"].apply(preprocess_text)

    text_df["text"].replace("", np.nan, inplace=True)
    text_df.dropna(subset=["text"], inplace=True)

    text_df.to_hdf("geo.h5", "text_df")

if __name__ == "__main__":
    main()
