import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from DecisionTree import DecisionTree


df = pd.read_csv('Thyroid_Diff.csv')

encoding = {}

attributes = df.columns

for attr in attributes:
    encoding[attr]  = {}


for i, row in df.iterrows():
    for j in range(1, len(row)):
        if not df.iloc[i, j] in encoding[attributes[j]]:
            encoding[attributes[j]][df.iloc[i, j]] = len(encoding[attributes[j]])

print(f"Dimension of the dataset {df.shape}")
print(df.columns)

# df = df.replace(encoding)

X = df.drop('Recurred', axis=1)
Y = df['Recurred']

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# print(X_test.shape, y_test.shape) 

# tree = DecisionTree()
# tree.fit(X_train, y_train) 

# y_pred = tree.predict(X_test)

# none = 0
# yes = 0 
# no = 0
# for pred in y_pred:
#     if pred == "Yes":
#         yes += 1 
#     elif pred == "No":
#         no += 1
#     else:
#         none += 1

# print(none, yes, no )
# print(f"accuracy of {tree.accuracy(y_pred, y_test)}")

# tree.draw_tree()

# Using sklearn 
kf = KFold(n_splits=5, shuffle=True, random_state=4) 

for i,(train_index, test_index) in enumerate(kf.split(X)):
    X_train, Y_train = X.iloc[train_index], Y.iloc[train_index] 
    X_test, Y_test = X.iloc[test_index], Y.iloc[test_index] 

    # clf = DecisionTreeClassifier(max_depth=5)
    # clf.fit(X_train, Y_train)

    # y_pred = clf.predict(X_test)

    # accuracy = accuracy_score(y_pred, Y_test) 

    clf = DecisionTree(max_depth=3)
    print(f"\nTRAINING FOR FOLD {i}")
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)

    accuracy = clf.accuracy(y_pred, Y_test)

    print(f"fold {i} accuracy= {accuracy*100}") 
    clf.draw_tree(f"plots/fold_{i}.png")

    # plt.figure(figsize=(15, 10))
    # plot_tree(clf, filled=True)
    # plt.show()