# Deep Text Eval (DTE)

## Problem
A differentiable function that measures text complexity/readability will be beneficial as part of the loss function of neural network text simplification system. However, to our best knowledge, there are not such a differentiable
function.

## Basic Idea
Train an RNN to measure the ext complexity/readability. RNN is differentiable by
its nature. This RNN will be part of a future building block for the loss of a
neural network text simplification system.

## Learning Task
Leveraging a classification of news article by
[CEFR](https://en.wikipedia.org/wiki/Common_European_Framework_of_Reference_for_Languages) into a real number readability score.

## Corpus
[Deutsche Welle - Deutsche Lernen](http://www.dw.com/de/deutsch-lernen/s-2055) news articles.


### Readability Levels
```
0 - B1
1 - B2, C1
```

### Structure
In the HD5 file `data/dw.h5` there are three keys holding 3 pandas' data frames:

1. `pages_df` - Web pages in DW website.
2. `text_df` - News articles.
3. `paragraphs_df`- Paragraphs of news articles.

The important columns of `text_df` and `paragraphs_df` are:
1. `url` - URL of the article page.
2. `title` - The tile of the news article.
3. `text` - The text itself.
4. `y` - The level label (`0` or `1`), as described above.

### Use in Python
```python
import pandas as pd

with pd.HDFStore('data/dw.h5') as dw_store:
	pages_df = dw_store["pages_df"]
	text_df = dw_store["text_df"]
	paragraphs_df = dw_store["paragraphs_df"]
```
