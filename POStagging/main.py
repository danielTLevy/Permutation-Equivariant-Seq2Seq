from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

from collections import Counter

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

from nltk.stem.snowball import FrenchStemmer
from nltk.tag import StanfordPOSTagger
jar = './stanford-postagger-full-2020-11-17/stanford-postagger-4.2.0.jar'
french_model = './stanford-postagger-full-2020-11-17/models/french-ud.tagger'
import os
java_path = "/usr/bin/java"
os.environ['JAVAHOME'] = java_path
french_pos_tagger = StanfordPOSTagger(french_model, jar, encoding='utf8' )

######################################################################
# We'll need a unique index per word to use as the inputs and targets of
# the networks later. To keep track of all this we will use a helper class
# called ``Lang`` which has word → index (``word2index``) and index → word
# (``index2word``) dictionaries, as well as a count of each word
# ``word2count`` to use to later replace rare words.
#

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################
# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, make everything lowercase, and trim most
# punctuation.
#

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


######################################################################
# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.
#

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


######################################################################
# Since there are a *lot* of example sentences and we want to train
# something quickly, we'll trim the data set to only relatively short and
# simple sentences. Here the maximum length is 10 words (that includes
# ending punctuation) and we're filtering to sentences that translate to
# the form "I am" or "He is" etc. (accounting for apostrophes replaced
# earlier).
#

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

lemmatizer = WordNetLemmatizer()
lemmatize = True

input_pos = []
output_pos = []
input_pos_1_verb = []
input_lemmatized_pos_1_verb = []

all_eng_verbs = Counter()
single_eng_verbs = Counter()

single_verb_pairs = []

for [fre, eng] in pairs:
    eng_pos = (pos_tag((word_tokenize(eng))))
    fre_pos = french_pos_tagger.tag(word_tokenize(fre,language='french'))
    print(fre_pos)
    input_pos.append(eng_pos)
    output_pos.append(fre_pos)
    eng_verbs = []
    lemmatized_eng = []
    for word, tag in eng_pos:
        tag = tag[0].lower()
        if lemmatize:
            if tag in ['v', 'a', 'r', 'n']:
                word = lemmatizer.lemmatize(word,tag)
            else:
                word = lemmatizer.lemmatize(word)
            lemmatized_eng.append(word)
        if tag in ['v', 'm']:
            eng_verbs.append(word)  # get all the verbs in the sentence
            all_eng_verbs[word] += 1  # keeping track of all the verbs in the whole corpus

    fre_verbs = []
    for word, tag in fre_pos:
       if tag in ['VERB','AUX']:
           fre_verbs.append(word)

    if (len(eng_verbs) == 1):  # if theres one verb in the sentence
        single_eng_verbs[eng_verbs[0]] += 1  # keeping track of all the sentenes
        input_pos_1_verb.append(eng)
        # input_pos_1_verb.append(eng_pos)
        if lemmatize:
            input_lemmatized_pos_1_verb.append(lemmatized_eng)

    if len(eng_verbs)==1 and len(fre_verbs) == 1:
       single_verb_pairs.append((eng,fre))



print(len(input_pos_1_verb))
print(Counter(all_eng_verbs).most_common(50))
print(single_eng_verbs)
print(len(input_lemmatized_pos_1_verb))

print(input_pos_1_verb[5:30])
print(input_lemmatized_pos_1_verb[5:30])

print(output_pos)
print((single_verb_pairs))
print(len(input_pos_1_verb))
print(len(single_verb_pairs))
