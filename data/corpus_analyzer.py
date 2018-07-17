"""
A corpus should be in a pandas dataframe with two columns:
* `text`
    * Single space as words seperator
    * `.` as sentences seperator
    * Single \n` as paragraphs seperator
* `y`
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pylab as plt
import matplotlib.patches as mpatches
import seaborn as sns

from scipy.stats import zscore

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD, PCA, LatentDirichletAllocation
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from textstat.textstat import textstat

COUNT_TYPES = ["wc", "sc", "pc"]

plt.ion()


def generate_color_numbers_dict(series):
    labels = series.unique()
    labels.sort()
    return {label:"C" + str(color)
            for color, label in enumerate(labels)}


def plot_y_count(df):
    sns.countplot(df["y"])


def enrich_with_counts(df):
    df["wc"] = df["text"].apply(lambda x: len(x.split()))
    # split by ".\s" or ".$"
    df["sc"] = df["text"].apply(lambda x: len(x.split(".")))
    df["pc"] = df["text"].apply(lambda x: len(x.split("\n")))

    return df


def get_all_counts_stats(print_all_counts_stats):
    counts_sum = print_all_counts_stats[COUNT_TYPES].agg(sum)
    counts_sum = counts_sum.rename("sum")
    counts_stats = print_all_counts_stats[COUNT_TYPES].describe()
    counts_stats = counts_stats.append(counts_sum)
    return counts_stats


def plot_count_dist(df, count_type, ax=None):
    if ax is None:
        f, ax = plt.subplots(1)
        ax.legend()
    ax.set_title(count_type)

    for y in df["y"].unique():
        sns.distplot(df[df["y"] == y][count_type], label=str(y), ax=ax)


def plot_all_counts_dist(df):
    f, axes = plt.subplots(1, len(COUNT_TYPES), figsize=(5*len(COUNT_TYPES),5))
    f.tight_layout()
    for count_type, ax in zip(COUNT_TYPES, axes):
        plot_count_dist(df, count_type, ax)
    axes[-1].legend()


def get_lognest_words(df, n=10):
    return ((df["text"]
            .apply(lambda text:
                max(zip(map(len, text.split()), text.split()))))
            .sort_values()
            .tail(n))


def plot_FKG(df):
    "https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests"

    f, ax = plt.subplots(1)
    df["FKG"] = df["text"].apply(textstat.flesch_kincaid_grade)

    for y in sorted(df["y"].unique()):
        sns.distplot(df[df["y"] == y]["FKG"], label=str(y), ax=ax)

    ax.legend()
    ax.set_title("Distribution Flesch-Kincaid Grade per Text by Reading Lables")


def to_array(vectors):
    if not isinstance(vectors, np.ndarray):
        vectors = vectors.toarray()
    return vectors


def exttract_td_idf_features(df, no_features=1000):
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf_vectors = tf_vectorizer.fit_transform(df["text"])
    tf_feature_names = tf_vectorizer.get_feature_names()
    return tf_vectors, tf_feature_names


def plot_t_SNE(df, vectors, normalized=False, reduced_dim=50, perplexity=40, title=""):
    if reduced_dim is not None:
        X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(vectors)
    else:
        X_reduced = to_array(vectors)

    if normalized:
        X_reduced = zscore(to_array(X_reduced), axis=0)

    X_embedded = TSNE(n_components=2, perplexity=40, verbose=0).fit_transform(X_reduced)

    colors_dict = generate_color_numbers_dict(df["y"])

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(frameon=False)
    plt.setp(ax, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                    wspace=0.0, hspace=0.0)
    ax.set_title("t-SNE " + title)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
        c=df["y"].map(colors_dict), marker="x",
        label=df["y"].map(colors_dict))

    plt.legend(handles=[mpatches.Patch(color=color, label=label)
        for label, color in colors_dict.items()])


def plot_PCA(df, vectors, title=""):
    X_embedded = PCA(n_components=2, random_state=0).fit_transform(to_array(vectors))

    colors_dict = generate_color_numbers_dict(df["y"])

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(frameon=False)
    plt.setp(ax, xticks=(), yticks=())
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                    wspace=0.0, hspace=0.0)
    ax.set_title("PCA " + title)
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1],
            c=df["y"].map(colors_dict), marker="x",
            label=df["y"].map(colors_dict))

    plt.legend(handles=[mpatches.Patch(color=color, label=label)
        for label, color in colors_dict.items()])


def classify_SVM(df, vectors):
    X_train, X_test, y_train, y_test = train_test_split(
        to_array(vectors), df["y"], test_size=0.2, random_state=42)

    clf_svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=0)
    clf_svm.fit(X_train, y_train)
    
    y_pred = clf_svm.predict(X_test)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    
    return classification_report(y_test, y_pred) + '\nAccuracy = {:0.3f}'.format(accuracy)


def train_lda(tf_vectors, n_topics=20):
    lda = LatentDirichletAllocation(n_topics=n_topics,
            max_iter=5,
            learning_method='online',
            learning_offset=50.,
            random_state=0,
            verbose=False).fit(tf_vectors)

    doc_topics_vectors = lda.transform(tf_vectors)

    return lda, doc_topics_vectors


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


def generate_descriptive_analysis(df):
    plot_y_count(df)
    df = enrich_with_counts(df)
    print(get_all_counts_stats(df))
    print()
    plot_all_counts_dist(df)
    print("Longest Words: ", get_lognest_words(df))
    print()
    plot_FKG(df)


def generate_tf_idf_analysis(df, tf_vectors, tf_feature_names):
    plot_PCA(df, tf_vectors, title="TF-IDF")
    plot_t_SNE(df, tf_vectors, title="TF-IDF Not-Normalized Embedded")    
    plot_t_SNE(df, tf_vectors, reduced_dim=None, title="TF-IDF Not-Normalized Not-Embedded")    
    plot_t_SNE(df, tf_vectors, normalized=True, title="TF-IDF Normalized Embedded")
    plot_t_SNE(df, tf_vectors, normalized=True, reduced_dim=None, title="TF-IDF Normalized Not-Embedded")    
    print("SVM - TF-IDF", classify_SVM(df, tf_vectors))
    print()


def generate_lda_analysis(df, tf_vectors, tf_feature_names):
    lda, doc_topics_vectors = train_lda(tf_vectors, n_topics=20)
    plot_PCA(df, doc_topics_vectors, title="LDA")
    plot_t_SNE(df, doc_topics_vectors, reduced_dim=None, title="LDA")
    plot_t_SNE(df, doc_topics_vectors, normalized=True, reduced_dim=None, title="LDA Normalized")
    
    print("SVM - LDA", classify_SVM(df, doc_topics_vectors))
    print()


def generate_features_analysis(df):
        tf_vectors, tf_feature_names = exttract_td_idf_features(df)

        generate_tf_idf_analysis(df, tf_vectors, tf_feature_names)
        print()

        generate_lda_analysis(df, tf_vectors, tf_feature_names)
        print()


def generate_corpus_analysis(df):
    generate_descriptive_analysis(df)
    print()
    generate_features_analysis(df)
