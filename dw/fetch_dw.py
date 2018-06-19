"""
TODO
[x] seperate intro & text
[x] filter out intro & text
[x] Lektion checking and new artickle
[x] change data to page_df
[ ] split LANGSAM GESPROCHENE NACHRICHTEN to articles - debug and validate `build_text_df`
[x] seperate script 
[ ] documentation
[ ] run it all
[ ] notebook for corpus analysis
"""

import time
import warnings

import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import requests
import tqdm
from tqdm import tqdm
from bs4 import BeautifulSoup

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import grequests


tqdm.pandas()


URL_SEARCH_FORMAT = "http://www.dw.com/search/?{item_field}{content_type_field}languageCode=de&searchNavigationId={rubrik}&to={to}&sort=DATE&resultsCounter={counter}"

DW_RUBRIK = {"THEMEN": 9077, "DEUTSCH LERNEN": 2055}

LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

NUM_PARALLEL_REQUESTS = 5

ARTIKELS = {"DEUTSCH LERNEN": {"Nachrichten",                      # B2 & C1 - Multiple Text
                               "Langsam gesprochene Nachrichten",  # B2 & C1 - Multiple Text
                               "Top-Thema – Podcast"},             # B1 - One Text
            "LEKTIONEN": {"Top-Thema – Lektionen"},                # B1 - One Text
           }

TEXT_DF_COLUMNS = ["url", "rubrik", "y"]


def generate_dw_search_url(rubrik, counter=1000, to=None, **kwargs):
    assert rubrik in DW_RUBRIK
    if rubrik == "DEUTSCH LERNEN":
        assert "item" in kwargs and kwargs["item"] in LEVELS

    if to is None:
        to = time.strftime("%d.%m.%Y")

    format_kwargs = {"rubrik": DW_RUBRIK[rubrik],
                    "counter": counter,
                    "to": to}

    if rubrik == "THEMEN":
        format_kwargs["content_type_field"] = "contentType=ARTICLE&"
    else:
        format_kwargs["content_type_field"] = ""

    if "item" in kwargs:
        format_kwargs["item_field"] = "item={}&".format(kwargs["item"])
    else:
        format_kwargs["item_field"] = ""

    return URL_SEARCH_FORMAT.format(**format_kwargs)


def get_dw_urls(rubrik, limit=np.inf, **kwargs):

    urls = set()
    to = None
    keep_scraping = True

    while keep_scraping:
        dw_url = generate_dw_search_url(rubrik=rubrik,
                                        to=to,
                                        **kwargs)

        r = requests.get(dw_url)

        soup = BeautifulSoup(r.content, "html.parser")

        search_results = soup.find_all(class_="searchResult")
        if search_results:
            urls |= {result.a["href"] for result in search_results}
            new_to = soup.find_all("span", class_="date")[-1].get_text()
        else:
            keep_scraping = False

        if len(urls) >= limit:
            keep_scraping = False

        if new_to == to:
            keep_scraping = False
        else:
            to = new_to

    return urls


def build_initial_pages_df(rubrik, urls):
    return pd.DataFrame({"rubrik": rubrik, "url": urls})

def initialize_pages_df():
    print("Retrieving all articles URLS...")
    pages_df = pd.concat([build_initial_pages_df("DEUTSCH LERNEN",
                                    list(set.union(*(get_dw_urls("DEUTSCH LERNEN", item=level)
                                                 for level in tqdm(LEVELS)))))])

                                                 #,
                     # pd.DataFrame({"rubrik": "THEMEN",
                     #               "url": list(get_dw_urls("THEMEN",
                     #                                       limit=2000))})


    return pages_df


def batches(iterable, n=1):
    """
    From http://stackoverflow.com/a/8290508/270334
    :param n:
    :param iterable:
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def exception_handler(request, exception):
    print("{} failed: {} ".format(request.kwargs, exception))


def fetch_html(df):

    print("Retrieving all the pages...")
    n_iters = len(df)//NUM_PARALLEL_REQUESTS + 1
    failed_requests = []
    rs = []

    for batch in tqdm(batches(df["url"], NUM_PARALLEL_REQUESTS), total=n_iters):
        requests_list = [grequests.get("http://www.dw.com" + url, stream=False) for url in batch]
        responses = grequests.map(requests_list, exception_handler=exception_handler)

        for response in responses:
            if response is None:
                continue
            if response.status_code != 200:
                failed_requests.append(response.text)

        rs.extend(responses)


    df["request"] = rs

    print("Second Downloading for failed ones...")
    df["request"] = df.apply(lambda r:
                                    r["request"]
                                    if r["request"] is not None
                                    else requests.get("http://dw.com" + r["url"]),
                                  axis=1)

    print("Extracting HTML...")
    df["html"] = df.apply(lambda r:
                                    r["request"].content
                                    if r["request"].content.strip().endswith(b"</html>")
                                    else requests.get("http://dw.com" + r["url"]).content,
                                  axis=1)


    is_full_html_mask = df["html"].str.strip().str.endswith(b"</html>")
    df = df[is_full_html_mask]

    print("Count of articles withh full HTML...")
    print(is_full_html_mask.value_counts())

    return df


def soupify(df):
    print("Instantiating soup objects...")
    df["soup"] = df["html"].progress_apply(lambda html:
                                               BeautifulSoup(html, "html.parser"))
    return df

def enrich_with_lectionen_pages_df(df):
    print("Enriching with LEKTIONEN pages...")
    lektion = df["soup"].apply(lambda soup: soup.find("span", string="Lektion"))

    df = df[lektion.isnull()]
    lektion = lektion.dropna()

    lektion_pages_df = build_initial_pages_df("LEKTIONEN",
                            list(lektion.apply(lambda tag: tag.parent.parent["href"])))
    lektion_pages_df = fetch_html(lektion_pages_df)
    lektion_pages_df = soupify(lektion_pages_df)

    return pd.concat([df, lektion_pages_df])

def extract_artikel(df):
    print("Extracting artikel type...")
    df["artikel"] = df["soup"].apply(lambda soup:
                                             soup.find(class_="artikel").get_text())
    return df


def filter_by_artikel(df):
    print("Filtering by artikel...")
    df = df[(df["rubrik"] == "THEMEN")
                  | ((df["rubrik"] == "DEUTSCH LERNEN") & (df["artikel"].isin(ARTIKELS["DEUTSCH LERNEN"])))
                  | ((df["rubrik"] == "LEKTIONEN") & (df["artikel"].isin(ARTIKELS["LEKTIONEN"])))]
    return df


def extract_meta(df):
    """
    print("Extracting titles...")
    df["title"] = df["soup"].apply(lambda soup:
                                         soup.h1.get_text())
    """
    
    print("Extracting tags...")
    df["tags"]  =  df["soup"].progress_apply(lambda soup:
                        [tag.text for tag in soup
                                                 .find("ul", class_="smallList")
                                                 .find_all("a")
                             if tag["href"].startswith("/search/")])

    print("Extracting levels...")
    df["levels"] = df["tags"].progress_apply(lambda tags:
                                            tuple(level for level in LEVELS if level in tags))

    return df


def encode_level(r):
    if r["rubrik"] in {"DEUTSCH LERNEN", "LEKTIONEN"}:
        if r["levels"] == ("B1",):
            return 0
        elif r["levels"] == ("B2", "C1"):
            return 1
        else:
            return np.nan
    elif r["rubrik"] == "THEMEN":
        return 2
    else:
        return np.nan

def encode_level_labels(df):
    print("Encoding level labels...")
    df["y"] = df.progress_apply(encode_level, axis=1)
    return df


def extract_intro(soup):
    try:
        intro = soup.find("p", class_="intro").getText().strip()
    except AttributeError:
        return np.nan
    else:
        if not intro:
            return np.nan
        else:
            return intro

        
 ######### TODO: REFACTOR ALL THIS PART #########
def paragraphy(tag):
    return "\n".join([paragraph.get_text().strip() for paragraph in tag.find_all("p")])

def extract_content_DEUTSCH_LERNEN_SINGLE(soup):
    paragraphs = []
    long_text_tag = soup.find("div", class_="longText")
    for tag in long_text_tag.childGenerator():
        if tag.name == "p":
            paragraphs.append(tag.get_text().strip())
        elif tag.name == "br":
            break
    return "\n".join(paragraphs)
    #return paragraphy(soup.find("div", class_="longText")).split("Glossar", 1)[0].strip()

def extract_content_DEUTSCH_LERNEN_MULTIPLE(soup):
    return [tag.get_text().strip()
                for tag in soup.find("div", class_="longText").find_all("p")
                    if tag.find("strong") is None 
                        and tag.find("b") is None
                        and tag.getText().strip()
                        and not tag.getText().strip().startswith("***")]


def extract_content_THEMEN(soup):
    return paragraphy(soup.find("div", class_="longText"))

def extract_content_LEKTIONEN(soup):
    content_part = list(soup.find("div", class_="dkTaskWrapper tab3").children)[3]
    for definition_bubble in content_part.find_all("span"):
        definition_bubble.decompose()
    return paragraphy(content_part)

"""
def extract_content(r):
    try:
        if r["rubrik"] == "DEUTSCH LERNEN":
            content = extract_content_DEUTSCH_LERNEN(r["soup"])
        elif r["rubrik"] == "THEMEN":
            content = extract_content_THEMEN(r["soup"])
        elif r["rubrik"] == "LEKTIONEN":
            content = extract_content_LEKTIONEN(r["soup"])
    except AttributeError:
        return np.nan
    else:
        if not content:
            return np.nan
        else:
            return content

def extract_text(df):
    print("Extracting intro...")
    df["intro"] = df["soup"].progress_apply(extract_intro)
    print("Extracting content...")
    df["content"] = df.progress_apply(extract_content, axis=1)
    print("Building text...")
    df["text"] = df["intro"] + df["content"]

    return df
"""

def build_text_rows(page_row):
    text_row = page_row.copy()[TEXT_DF_COLUMNS]
    text_rows = []
    
    if page_row["rubrik"] == "DEUTSCH LERNEN":
    
        if page_row["artikel"] == "Top-Thema – Podcast":
            text_row["text"] = extract_content_DEUTSCH_LERNEN_SINGLE(page_row["soup"])
            text_rows = [text_row]

        elif page_row["artikel"] in {"Nachrichten", "Langsam gesprochene Nachrichten"}:
            for text in extract_content_DEUTSCH_LERNEN_MULTIPLE(page_row["soup"]):
                current_text_row = text_row.copy()
                current_text_row["text"] = text
                text_rows.append(current_text_row)
    
    elif page_row["rubrik"] == "LEKTIONEN" and page_row["artikel"] == "Top-Thema – Lektionen":
        text_row["text"] = extract_content_LEKTIONEN(page_row["soup"])
        text_rows = [text_row]
    
    elif page_row["rubrik"] == "THEMEN":
        text_row["text"] = extract_content_THEMEN(page_row["soup"])
        text_rows = [text_row]
    
    return pd.DataFrame(text_rows)

 ######### END TODO: REFACTOR ALL THIS PART #########


def handle_article_withot_content(df):
    article_withot_content_mask = df["content"].isnull()

    if article_withot_content_mask.any():

        print("Articles without content:")
        for url in df[article_withot_content_mask]["url"]:
            print(url)

        print("Dropping articles without content...")
        df = df[~article_withot_content_mask]

    return df

def build_pages_df(n_pages=None, to_filter=True):
    pages_df = initialize_pages_df()
    
    if n_pages is not None:
        pages_df = pages_df.sample(n_pages)

    pages_df = fetch_html(pages_df)
    pages_df = soupify(pages_df)

    pages_df = enrich_with_lectionen_pages_df(pages_df)

    pages_df = extract_artikel(pages_df)

    if to_filter:
        pages_df = filter_by_artikel(pages_df)
    
    pages_df = extract_meta(pages_df)

    pages_df = encode_level_labels(pages_df)

    pages_df = pages_df.reset_index(drop=True)

    return pages_df

def build_text_df(pages_df):
    print("Bulding Text Dataframe...")
    text_rows = [build_text_rows(page_row)
                 for _, page_row in tqdm(pages_df.iterrows(), total=len(pages_df))]
    text_df = pd.concat(text_rows)
    return text_df

def main():
    pages_df = build_pages_df(500)
    text_df = build_text_df(pages_df)
#    text_df = handle_article_withot_content(text_df)

    print(text_df.head())

if __name__ == "__main__":
    main()
