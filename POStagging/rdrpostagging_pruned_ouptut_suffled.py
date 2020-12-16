from ImportData import pairs
from ImportRDSTagger import french_tagger, frenchDICT, english_tagger, englishDICT
from collections import Counter
import pickle
import os
from sklearn.model_selection import train_test_split
#from nltk.stem import WordNetLemmatizer
#lemmatizer = WordNetLemmatizer()

def read_tagged_sentences (tagger,dic,sentence):
    tagged_sentence = tagger.tagRawSentence(dic, sentence)
    verbs = []
    for word_tag in tagged_sentence.split():
        word, tag = word_tag.rsplit('/',1)
        if tag[0].upper() == 'N':
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

verb_pairs_pruined = Counter()

for ((eng_verb, fre_verb),frequency) in verb_pairs.most_common():
    if len(verb_pairs_pruined) >= 10:
        break
    if eng_verb not in in_equivariances and fre_verb not in out_equivariances: #ensure equivariances are unique
        in_equivariances.append(eng_verb)
        out_equivariances.append(fre_verb)
        verb_pairs_pruined[(eng_verb,fre_verb)]+=1
        #print((eng_verb,fre_verb),frequency)

print(verb_pairs_pruined)
print(len(in_equivariances))
print(len(out_equivariances))

english_in_pruned = []
french_out_pruned = []
for [fre, eng] in pairs:
    eng_verbs = read_tagged_sentences(english_tagger,englishDICT,eng)
    fre_verbs = read_tagged_sentences(french_tagger,frenchDICT,fre)
    if len(eng_verbs)==1 and len(fre_verbs) == 1 and verb_pairs_pruined[(eng_verbs[0],fre_verbs[0])]>0:
        english_in_pruned.append(eng)
        french_out_pruned.append(fre)

print(len(english_in_pruned))
print(len(french_out_pruned))

X_train, X_test, y_train, y_test = train_test_split(english_in_pruned, french_out_pruned, test_size=0.20, shuffle=True, random_state=42)

print(X_test)
print(y_test)

os.chdir("../../output")
with open('X_train.data', 'wb') as f:
    pickle.dump(X_train, f)
with open('X_test.data', 'wb') as f:
        pickle.dump(X_test, f)
with open('y_train.data', 'wb') as f:
    pickle.dump(y_train, f)
with open('y_test.data', 'wb') as f:
    pickle.dump(y_test, f)
'''
with open('equivariance_frequencies.data', 'wb') as f:
    pickle.dump(verb_pairs_pruined.most_common(), f)
'''
with open('in_equivariances.data', 'wb') as f:
    pickle.dump(in_equivariances, f)
with open('out_equivariances.data', 'wb') as f:
    pickle.dump(out_equivariances, f)
