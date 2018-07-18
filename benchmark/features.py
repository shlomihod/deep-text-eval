import itertools as it
from collections import Counter, deque

import numpy as np
import spacy
from spacy import displacy
from benepar.spacy_plugin import BeneparComponent
from tqdm import tqdm


def _count_iter_items(iterable):
    """
    Consume an iterable not reading it into memory; return the number of items.
    https://stackoverflow.com/questions/3345785/getting-number-of-elements-in-an-iterator-in-python
    """
    counter = it.count()
    deque(zip(iterable, counter), maxlen=0)  # (consume at C speed)
    return next(counter)


def _calc_depth(span):
    children = list(span._.children)
    if not children:
        return 0
    else:
        return max(_calc_depth(child) for child in children) + 1


def _count_constituent(doc):
    """Counts the constituents in a given document."""
    constituent_counter = Counter()
    for sent in doc.sents:
        for const in sent._.constituents:
            constituent_counter.update(Counter(const._.labels))

    return constituent_counter


def syntactic_complexity(doc):
    constituent_counter = _count_constituent(doc)
    n_sentences = _count_iter_items(doc.sents)
    return {
        'avgparsetreeheight': np.mean([_calc_depth(sent) for sent in doc.sents]),  # average height of a parse Tree
        'senlen': np.mean([len(sent) for sent in doc.sents]),     # average sentence length
        'numnp': constituent_counter['NP'] / n_sentences,         # noun phrases (NP) / sentences
        'numpp': constituent_counter['PP'] / n_sentences,         # prepositional phrases (PP) / sentences
        'numsbar': constituent_counter['SBAR'] / n_sentences,  # SBARs/sentences
    }


def celex_complexity(doc):
    constituent_counter = _count_constituent(doc)
    return {
        'numprep': constituent_counter["PP"],  # for every PP, there is exactly one preposition
    }


def tag_counts(doc):
    tag_counts = Counter([token.tag_ for token in doc])
    return {
        'num_-LRB-': tag_counts['-LRB-'], # left round bracket
        'num_-RRB-': tag_counts['-RRB-'], # right round bracket
        'num_,': tag_counts[','], # punctuation mark, comma
        'num_:': tag_counts[':'], # punctuation mark, colon or ellipsis
        'num_.': tag_counts['.'], # punctuation mark, sentence closer
        'num_''': tag_counts["''"], # closing quotation mark
        'num_""': tag_counts['""'], # closing quotation mark
        'num_#': tag_counts['#'], # symbol, number sign
        'num_``': tag_counts['``'], # opening quotation mark
        'num_$': tag_counts['$'], # symbol, currency
        'num_ADD': tag_counts['ADD'], # email
        'num_AFX': tag_counts['AFX'], # affix
        'num_BES': tag_counts['BES'], # auxiliary "be"
        'num_CC': tag_counts['CC'], # conjunction, coordinating
        'num_CD': tag_counts['CD'], # cardinal number
        'num_DT': tag_counts['DT'], # 
        'num_EX': tag_counts['EX'], # existential there
        'num_FW': tag_counts['FW'], # foreign word
        'num_GW': tag_counts['GW'], # additional word in multi-word expression
        'num_HVS': tag_counts['HVS'], # forms of "have"
        'num_HYPH': tag_counts['HYPH'], # punctuation mark, hyphen
        'num_IN': tag_counts['IN'], # conjunction, subordinating or preposition
        'num_JJ': tag_counts['JJ'], # adjective
        'num_JJR': tag_counts['JJR'], # adjective, comparative
        'num_JJS': tag_counts['JJS'], # adjective, superlative
        'num_LS': tag_counts['LS'], # list item marker
        'num_MD': tag_counts['MD'], # verb, modal auxiliary
        'num_NFP': tag_counts['NFP'], # superfluous punctuation
        'num_NIL': tag_counts['NIL'], # missing tag
        'num_NN': tag_counts['NN'], # noun, singular or mass
        'num_NNP': tag_counts['NNP'], # noun, proper singular
        'num_NNPS': tag_counts['NNPS'], # noun, proper plural
        'num_NNS': tag_counts['NNS'], # noun, plural
        'num_PDT': tag_counts['PDT'], # predeterminer
        'num_POS': tag_counts['POS'], # possessive ending
        'num_PRP': tag_counts['PRP'], # pronoun, personal
        'num_PRP$': tag_counts['PRP$'], # pronoun, possessive
        'num_RB': tag_counts['RB'], # adverb
        'num_RBR': tag_counts['RBR'], # adverb, comparative
        'num_RBS': tag_counts['RBS'], # adverb, superlative
        'num_RP': tag_counts['RP'], # adverb, particle
        'num__SP': tag_counts['_SP'], # space
        'num_SYM': tag_counts['SYM'], # symbol
        'num_TO': tag_counts['TO'], # infinitival to
        'num_UH': tag_counts['UH'], # interjection
        'num_VB': tag_counts['VB'], # verb, base form
        'num_VBD': tag_counts['VBD'], # verb, past tense
        'num_VBG': tag_counts['VBG'], # verb, gerund or present participle
        'num_VBN': tag_counts['VBN'], # verb, past participle
        'num_VBP': tag_counts['VBP'], # verb, non-3rd person singular present
        'num_VBZ': tag_counts['VBZ'], # verb, 3rd person singular present
        'num_WDT': tag_counts['WDT'], # wh-determiner
        'num_WP': tag_counts['WP'], # wh-pronoun, personal
        'num_WP$': tag_counts['WP$'], # wh-pronoun, possessive
        'num_WRB': tag_counts['WRB'], # wh-adverb
        'num_XX': tag_counts['XX'], # unknown
}

#######################################################

EXTRACTORS = [
    syntactic_complexity,
    celex_complexity,
    tag_counts
]


def extract_doc_features(doc):
    features = {}
    for extractor in EXTRACTORS:
        features.update(extractor(doc))
    doc._.features = features
    return doc

spacy.tokens.Doc.set_extension('features', default={}, force=True)
    
nlp = spacy.load('en', disable=['nre'])
nlp.add_pipe(BeneparComponent("benepar_en_small"))
nlp.add_pipe(extract_doc_features, name='extract_doc_features', first=False)

def docify_text(texts, batch_size=100, n_threads=4):
    return [doc for doc in nlp.pipe(tqdm(texts), batch_size=batch_size, n_threads=n_threads)]
