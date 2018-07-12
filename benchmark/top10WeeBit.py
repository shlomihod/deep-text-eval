import spacy
import numpy as np

test_string = """Max excels at large-scale information extraction tasks. It's written from the ground up in carefully memory-managed Cython. Independent research has confirmed that Max is the fastest in the world. If your application needs to process entire web dumps, Max is the library you want to be using."""
# test_string = "I go to my beautiful home."

# print and parse test string
nlp = spacy.load('en_core_web_sm')
doc = nlp(test_string)
print(doc)
print()

# feature 4: Average height of a parse Tree
def get_tree_depth(tree):
    if not tree["modifiers"]:
        return 0
    else:
        return max([get_tree_depth(mod) for mod in tree["modifiers"]])+1

tree_lengths = [get_tree_depth(sentence) for sentence in doc.print_tree()]
print('tree_lengths:', tree_lengths)
print('avg_tree_lengths:', np.mean(tree_lengths))
