from ImportData import pairs
from ImportRDSTagger import french_tagger, frenchDICT, english_tagger, englishDICT
from collections import Counter
import pickle
import os
#from nltk.stem import WordNetLemmatizer
#lemmatizer = WordNetLemmatizer()

def read_tagged_sentences (tagger,dic,sentence):
    tagged_sentence = tagger.tagRawSentence(dic, sentence)
    verbs = []
    for word_tag in tagged_sentence.split():
        word, tag = word_tag.rsplit('/',1)
        if tag[0].upper() == 'V':
            verbs.append(word)
    return(verbs)

english_in = []
french_out = []

verb_pairs = Counter()

for [fre, eng] in pairs:
    eng_verbs = read_tagged_sentences(english_tagger,englishDICT,eng)
    fre_verbs = read_tagged_sentences(french_tagger,frenchDICT,fre)
    if len(eng_verbs)==1 and len(fre_verbs) == 1:
        english_in.append(eng)
        french_out.append(fre)
        eng_verb = eng_verbs[0]
        fre_verb = fre_verbs[0]
        #in_equivariances.append(eng_verb)
        #out_equivariances.append(fre_verb)
        verb_pairs[(eng_verb,fre_verb)]+=1

print(verb_pairs.most_common(50))
print(len(english_in))
print(len(french_out))

in_equivariances = []
out_equivariances = []

print(verb_pairs.most_common())

for ((eng_verb, fre_verb),frequency) in verb_pairs.most_common():
    if eng_verb not in in_equivariances and fre_verb not in out_equivariances:
        in_equivariances.append(eng_verb)
        out_equivariances.append(fre_verb)
        print((eng_verb,fre_verb),frequency)

print(len(in_equivariances))
print(len(out_equivariances))

os.chdir("../../output")
with open('in.data', 'wb') as f:
    pickle.dump(english_in, f)
with open('out.data', 'wb') as f:
        pickle.dump(french_out, f)
with open('in_equivariances.data', 'wb') as f:
    pickle.dump(in_equivariances, f)
with open('out_equivariances.data', 'wb') as f:
    pickle.dump(out_equivariances, f)
