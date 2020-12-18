from ImportData import pairs
from sklearn.model_selection import train_test_split
import pickle

english_nouns = ["book","car","house"]
french_nouns = ["livre", "voiture", "maison"]

X = []
y = []

for [fre, eng] in pairs:
    if any(noun in eng for noun in english_nouns) and any(noun in fre for noun in french_nouns):
        X.append(eng)
        y.append(fre)

print(len(X))
print(len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,shuffle=True, random_state=42)

#print(X_test)
#print(y_test)

with open('output2/X_train.data', 'wb') as f:
    pickle.dump(X_train, f)
with open('output2/X_test.data', 'wb') as f:
    pickle.dump(X_test, f)
with open('output2/y_train.data', 'wb') as f:
    pickle.dump(y_train, f)
with open('output2/y_test.data', 'wb') as f:
    pickle.dump(y_test, f)