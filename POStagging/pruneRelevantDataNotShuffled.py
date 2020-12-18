from ImportData import pairs
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle

english_nouns = ["book","car","house"]
french_nouns = ["livre", "voiture", "maison"]

# train_nouns = ["tom","something","car","time","problem","everyone","house","door","friends"]
# test_nouns = ["book"]

X_train = []  # doesn't contain book
y_train = []  # doesn't contain book
X_test = []  # contains book
y_test = []  # contains book

for [fre, eng] in pairs:
    if ("house" in eng and "maison" in fre):
        X_test.append(eng)
        y_test.append(fre)
    elif any(noun in eng for noun in english_nouns) and any(noun in fre for noun in french_nouns):
        X_train.append(eng)
        y_train.append(fre)

print(X_test)
print(y_test)
print(X_train)
print(y_train)
print(len(X_test))
print(len(y_test))
print(len(X_train))
print(len(y_train))

with open('output3/X_train.data', 'wb') as f:
    pickle.dump(X_train, f)
with open('output3/X_test.data', 'wb') as f:
    pickle.dump(X_test, f)
with open('output3/y_train.data', 'wb') as f:
    pickle.dump(y_train, f)
with open('output3/y_test.data', 'wb') as f:
    pickle.dump(y_test, f)
'''
book_vocab = []
with open("input/book_tom_vocab","r") as f:
   book_vocab = f.read().splitlines()

book_vocab.append("car")

#print(X_test)

good_in_sentences = []
good_out_sentences = []
for i in range(len(X_test)):
    eng_sentence = X_test[i]
    fre_sentence = y_test[i]
    if all(word in book_vocab for word in word_tokenize(eng_sentence)):
        good_in_sentences.append(eng_sentence)
        good_out_sentences.append(fre_sentence)

with open('output7booktomcar/X_test.txt', 'w') as f:
    for item in good_in_sentences:
        f.write("%s\n" % item)
with open('output7booktomcar/y_test.txt', 'w') as f:
    for item in good_out_sentences:
        f.write("%s\n" % item)
with open('output7booktomcar/X_test.data', 'wb') as f:
    pickle.dump(good_in_sentences, f)
with open('output7booktomcar/y_test.data', 'wb') as f:
    pickle.dump(good_out_sentences, f)
with open('output5/in_book_tom.txt', 'w') as f:
    for item in good_in_sentences:
        f.write("%s\n" % item)
with open('output5/out_book_tom.txt', 'w') as f:
    for item in good_out_sentences:
        f.write("%s\n" % item)

word_vocab = Counter()
for sentence in X_test:
    for word in word_tokenize(sentence):
        word_vocab[word] += 1

for (word,frequency) in word_vocab.most_common():
    print(word)


print((X_test))
print((y_test))
print(len(X_train))
print(len(y_train))

with open('output3/X_train.data', 'wb') as f:
    pickle.dump(X_train, f)
with open('output3/X_test.data', 'wb') as f:
    pickle.dump(X_test, f)
with open('output3/y_train.data', 'wb') as f:
    pickle.dump(y_train, f)
with open('output3/y_test.data', 'wb') as f:
    pickle.dump(y_test, f)
    
    
    
 X_train, X_test, y_train, y_test = train_test_split(good_in_sentences, good_out_sentences, test_size=0.20,shuffle=True, random_state=42)

print(X_train)
with open('output6booktom/X_train.txt', 'w') as f:
    for item in X_train:
        f.write("%s\n" % item)
with open('output6booktom/X_test.txt', 'w') as f:
    for item in X_test:
        f.write("%s\n" % item)
with open('output6booktom/y_train.txt', 'w') as f:
    for item in y_train:
        f.write("%s\n" % item)
with open('output6booktom/y_test.txt', 'w') as f:
    for item in y_test:
        f.write("%s\n" % item)   
'''
