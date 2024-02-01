import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from DecisionTree import DecisionTree


df = pd.read_csv('./Thyroid_Diff.csv')

encoding = {}

attributes = df.columns

for attr in attributes:
    encoding[attr]  = {}


for i, row in df.iterrows():
    for j in range(1, len(row)):
        if not df.iloc[i, j] in encoding[attributes[j]]:
            encoding[attributes[j]][df.iloc[i, j]] = len(encoding[attributes[j]])
        # df.iloc[i, j] = encoding[df.columns[j]][row[j]]

print(df.shape)

# df = df.replace(encoding)

X = df.drop('Recurred', axis=1)
Y = df['Recurred']

print(X.columns[15])

tree = DecisionTree()
initial_entropy = tree.entropy(X, Y, {})

print(initial_entropy)
print(tree.information_gain(X, Y, {}, "Smoking", initial_entropy)) 

tree.build(X, Y, {})

# Using sklearn 
# kf = KFold(n_splits=5, shuffle=True, random_state=4) 

# for i,(train_index, test_index) in enumerate(kf.split(X)):
#     X_train, Y_train = X.iloc[train_index], Y.iloc[train_index] 
#     X_test, Y_test = X.iloc[test_index], Y.iloc[test_index] 

#     clf = DecisionTreeClassifier(max_depth=5)
#     clf.fit(X_train, Y_train)

#     y_pred = clf.predict(X_test)

#     accuracy = accuracy_score(y_pred, Y_test) 
#     print(f"fold {i} accuracy= {accuracy*100}") 

#     plt.figure(figsize=(15, 10))
#     plot_tree(clf, filled=True)
#     plt.show()



