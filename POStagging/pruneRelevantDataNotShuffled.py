from ImportData import pairs
import pickle

english_nouns = ["tom","something","book","car","time","problem","everyone","house","door","friends"]
french_nouns = ["tom","chose","livre", "voiture", "temps", "probleme", "monde", "maison", "porte", "amis"]

#train_nouns = ["tom","something","car","time","problem","everyone","house","door","friends"]
#test_nouns = ["book"]

X_train = [] # doesn't contain book
y_train = [] # doesn't contain book
X_test = [] # contains book
y_test = [] # contains book

for [fre, eng] in pairs:
    if "book" in eng and "livre" in fre:
        X_test.append(eng)
        y_test.append(fre)
    elif any(noun in eng for noun in english_nouns) and any(noun in fre for noun in french_nouns):
        X_train.append(eng)
        y_train.append(fre)

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