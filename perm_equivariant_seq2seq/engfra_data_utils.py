from __future__ import unicode_literals, print_function, division
import unicodedata
from io import open
import re
import os
import pickle

from perm_equivariant_seq2seq.symmetry_groups import LanguageInvariance
from perm_equivariant_seq2seq.language_utils import Language, InvariantLanguage, EquivariantLanguage


SOS_token = 0
EOS_token = 1

"""
    SCAN Data handling
"""

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    
# Lowercase, trim, and remove non-letter characters
    

def normalize_string_engfra(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_engfra_data(path):
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize_string_engfra(s) for s in l.split('\t')] for l in lines]
    return pairs

def read_train_and_test(path):
     X_train =  pickle.load(open(os.path.join(path, 'X_train.data'), 'rb'))
     y_train =  pickle.load(open(os.path.join(path, 'y_train.data'), 'rb'))
     X_test =  pickle.load(open(os.path.join(path, 'X_test.data'), 'rb'))
     y_test =  pickle.load(open(os.path.join(path, 'y_test.data'), 'rb'))
     training_pairs = list(map(list, zip(X_train, y_train)))
     test_pairs = list(map(list, zip(X_test, y_test)))
     return training_pairs, test_pairs

def get_invariant_engfra_languages(pairs):
    # Initialize language classes
    input_lang = Language('eng')
    output_lang = Language('fra')
    # Set-up languages
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    # Return languages (including invariant syntax)
    return input_lang, output_lang


def get_equivariant_engfra_languages(pairs, input_equivariances, output_equivariances):
    # Initialize language classes
    input_lang = EquivariantLanguage('eng', input_equivariances)
    output_lang = EquivariantLanguage('fra', output_equivariances)
    # Set up languages
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    # Manipulate indices
    input_lang.rearrange_indices()
    output_lang.rearrange_indices()
    return input_lang, output_lang


def get_engfra_split(split=None):
    assert split in ['simple', 'add_book'], \
        "Please choose valid experiment split"
    DATA_DIR = 'POStagging/'

    # Simple (non-generalization) split
    if split == 'simple':
        data_path = os.path.join(DATA_DIR, 'ouputNounsShuffled')
    elif split == 'add_book':
        data_path = os.path.join(DATA_DIR, 'outputNounsSplitByBook')

    # Load data
    #all_pairs = read_engfra_data(data_path)
    training_pairs, test_pairs = read_train_and_test(data_path)
    print("Training pairs: ", len(training_pairs))
    print("Test pairs: ", len(test_pairs))
    return training_pairs, test_pairs


if __name__ == '__main__':
    train_pairs, test_pairs = get_engfra_split('simple')
