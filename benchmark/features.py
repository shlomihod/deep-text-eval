#!/usr/bin/python3
    
import itertools as it
from collections import Counter, deque, defaultdict

import numpy as np
import pandas as pd
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


def _calc_height(span):
    children = list(span._.children)
    if not children:
        return 0
    else:
        return max(_calc_height(child) for child in children) + 1


def _extract_constituent(doc):
    """Counts the constituents in a given document."""
    constituent_counter = Counter()
    constituent_lens = defaultdict(list)
    for sent in doc.sents:
        for const in sent._.constituents:
            constituent_counter.update(Counter(const._.labels))
            for label in const._.labels:
                constituent_lens[label].append(len(const))
    
    return constituent_counter, constituent_lens


def syntactic_complexity(doc):
    constituent_counter, constituent_lens = _extract_constituent(doc)
    n_sentences = _count_iter_items(doc.sents)
    return {
        
        'senlen': np.mean([len(sent) for sent in doc.sents]),     # average sentence length
        
        'numnp': constituent_counter['NP'] / n_sentences,         # NPs / sentences
        'numpp': constituent_counter['PP'] / n_sentences,         # PPs / sentences
        'numvp': constituent_counter['VP'] / n_sentences,         # VPs/sentences	
        'numsbar': constituent_counter['SBAR'] / n_sentences,  # SBARs/sentences
        'numsbarq': constituent_counter['SBARQ'] / n_sentences,  # SBARQs/sentences (questions?)
        'numwh': np.sum([constituent_counter[label] for label in constituent_counter
                            if label.startswith('WH')]) / n_sentences,  # WHs/sentences 
        
        'avgnpsize': np.mean(constituent_lens['NP']),  # average length of an NP
        'avgvpsize': np.mean(constituent_lens['VP']),  # average length of an VP
        'avgppsize': np.mean(constituent_lens['PP']),  # average length of an PP
        
        	#wh-phrases/sentences	 
        'avgparsetreeheight': np.mean([_calc_height(sent) for sent in doc.sents]),  # average height of a parse Tree

        'numconstituents': sum(constituent_counter.values()) / n_sentences, #constituents/sentences	
        
    }


def celex_complexity(doc):
    # TODO: why do we call `_extract_constituent` twice?
    constituent_counter, _ = _extract_constituent(doc)
    return {
        'numprep': constituent_counter["PP"],  # for every PP, there is exactly one preposition
    }


def pos_density(doc):
    tag_counts = Counter([token.tag_ for token in doc])
    pos_counts = Counter([token.pos_ for token in doc])
    
    n_sentences = _count_iter_items(doc.sents)

    # we use tokens and not word as in the phd, maybe we should change it
    n_tokens = sum([len(sent) for sent in doc.sents])
    
    return {
        'num_comma': tag_counts[','] / n_sentences, # punctuation mark, comma / sentences
        'nouns': (pos_counts['NOUN'] + pos_counts['PROPN']) / n_tokens, # (nouns + proper nouns)/all words
        'propernouns': pos_counts['PROPN'] / n_tokens, # proper nouns/all words
        'pronouns': (pos_counts['PRP'] + pos_counts['PRP$']) / n_tokens, # pronouns/all words
        
        # should we use ADP here?!?!
        'conj': (pos_counts['CONJ'] + pos_counts['ADP']) / n_tokens,  # conjunctions/all words	
        
        'adj': pos_counts['ADJ'] / n_tokens, # adjectives/all words	
        'ver': (pos_counts['VERB'] - tag_counts['MD']) / n_tokens, # non-modal verbs/all words
        'interj': pos_counts['INTJ'] / n_sentences, # interjections/total sentences
        'adverbs': pos_counts['ADV'] / n_sentences, # adverbs/total sentences
        'modals': tag_counts['MD'] / n_sentences, # modal verbs/total sentences
        'perpro': pos_counts['PRP'] / n_sentences, # personal pronouns/total sentences
        'whpro': (tag_counts['WP'] + tag_counts['WP$']) / n_sentences, # wh- pronouns/total sentences
        'numfuncwords': (tag_counts['BES'] + tag_counts['CC']
                        + tag_counts['DT'] + tag_counts['EX']
                        + tag_counts['HVS'] + tag_counts['IN']
                        + tag_counts['MD']	+ tag_counts['PRP']
                        + tag_counts['PRP$'] + tag_counts['RP']
                        + tag_counts['TO'] + tag_counts['UH']) / n_sentences, # function words/total sentences
        'numdet' : pos_counts['DET'] / n_sentences, # determiners/total sentences
        'numvb'  : tag_counts['VB'] / n_sentences,  # VB tags/total sentences
        'numvbd' : tag_counts['VBD'] / n_sentences, # VBD tags/total sentences
        'numvbg' : tag_counts['VBG'] / n_sentences, # VBG tags/total sentences
        'numvbn' : tag_counts['VBN'] / n_sentences, # VBN tags/total sentences
        'numvbp' : tag_counts['VBP'] / n_sentences, # VBP tags/total sentences
 	}	
        
        # 'num_-LRB-': tag_counts['-LRB-'], # left round bracket
        # 'num_-RRB-': tag_counts['-RRB-'], # right round bracket

        # 'num_:': tag_counts[':'], # punctuation mark, colon or ellipsis
        # 'num_.': tag_counts['.'], # punctuation mark, sentence closer
        # 'num_''': tag_counts["''"], # closing quotation mark
        # 'num_""': tag_counts['""'], # closing quotation mark
        # 'num_#': tag_counts['#'], # symbol, number sign
        # 'num_``': tag_counts['``'], # opening quotation mark
        # 'num_$': tag_counts['$'], # symbol, currency
        # 'num_ADD': tag_counts['ADD'], # email
        # 'num_AFX': tag_counts['AFX'], # affix
        # 'num_BES': tag_counts['BES'], # auxiliary "be"
        # 'num_CC': tag_counts['CC'], # conjunction, coordinating
        # 'num_CD': tag_counts['CD'], # cardinal number
        # 'num_DT': tag_counts['DT'], # 
        # 'num_EX': tag_counts['EX'], # existential there
        # 'num_FW': tag_counts['FW'], # foreign word
        # 'num_GW': tag_counts['GW'], # additional word in multi-word expression
        # 'num_HVS': tag_counts['HVS'], # forms of "have"
        # 'num_HYPH': tag_counts['HYPH'], # punctuation mark, hyphen
        # 'num_IN': tag_counts['IN'], # conjunction, subordinating or preposition
        # 'num_JJ': tag_counts['JJ'], # adjective
        # 'num_JJR': tag_counts['JJR'], # adjective, comparative
        # 'num_JJS': tag_counts['JJS'], # adjective, superlative
        # 'num_LS': tag_counts['LS'], # list item marker
        # 'num_MD': tag_counts['MD'], # verb, modal auxiliary
        # 'num_NFP': tag_counts['NFP'], # superfluous punctuation
        # 'num_NIL': tag_counts['NIL'], # missing tag
        # 'num_NN': tag_counts['NN'], # noun, singular or mass
        # 'num_NNP': tag_counts['NNP'], # noun, proper singular
        # 'num_NNPS': tag_counts['NNPS'], # noun, proper plural
        # 'num_NNS': tag_counts['NNS'], # noun, plural
        # 'num_PDT': tag_counts['PDT'], # predeterminer
        # 'num_POS': tag_counts['POS'], # possessive ending
        # 'num_PRP': tag_counts['PRP'], # pronoun, personal
        # 'num_PRP$': tag_counts['PRP$'], # pronoun, possessive
        # 'num_RB': tag_counts['RB'], # adverb
        # 'num_RBR': tag_counts['RBR'], # adverb, comparative
        # 'num_RBS': tag_counts['RBS'], # adverb, superlative
        # 'num_RP': tag_counts['RP'], # adverb, particle
        # 'num__SP': tag_counts['_SP'], # space
        # 'num_SYM': tag_counts['SYM'], # symbol
        # 'num_TO': tag_counts['TO'], # infinitival to
        # 'num_UH': tag_counts['UH'], # interjection
        # 'num_VB': tag_counts['VB'], # verb, base form
        # 'num_VBD': tag_counts['VBD'], # verb, past tense
        # 'num_VBG': tag_counts['VBG'], # verb, gerund or present participle
        # 'num_VBN': tag_counts['VBN'], # verb, past participle
        # 'num_VBP': tag_counts['VBP'], # verb, non-3rd person singular present
        # 'num_VBZ': tag_counts['VBZ'], # verb, 3rd person singular present
        # 'num_WDT': tag_counts['WDT'], # wh-determiner
        # 'num_WP': tag_counts['WP'], # wh-pronoun, personal
        # 'num_WP$': tag_counts['WP$'], # wh-pronoun, possessive
        # 'num_WRB': tag_counts['WRB'], # wh-adverb
        # 'num_XX': tag_counts['XX'], # unknown


#######################################################

EXTRACTORS = [
    syntactic_complexity,
    celex_complexity,
    pos_density
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


def main(path):
    print('Reading DFs...')
    with pd.HDFStore(path) as store:
        text_df = store['text_df']
        train_df = store['train_df']
        test_df = store['test_df']
    
    print('Extracting features...')
    docs = features.docify_text(text_df['text'])
    features_df = pd.DataFrame([doc._.features for doc in docs],
                                 index=text_df.index)
    features_df.columns = 'featues_' + features_df.columns

    print('Merging features...')
    text_features_df = text_df.merge(features_df, how='inner', left_index=True, right_index=True)
    train_features_df = train_df.merge(features_df, how='inner', left_index=True, right_index=True)
    test_features_df = test_df.merge(features_df, how='inner', left_index=True, right_index=True)

    assert len(text_df) == len(text_features_df)
    assert len(train_df) == len(test_features_df)
    assert len(test_df) == len(test_features_df)
    
    print('Saving DFs...')
    with pd.HDFStore(path) as store:
        text_features_df = store['text_features_df']
        train_features_df = store['train_features_df']
        test_features_df = store['test_features_df']

    
if __name__ == '__main__':
    import plac; plac.call(main)