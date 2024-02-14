# Group Number: 32
# 21CS10033 : Ishan Ray
# 20EE10089 : Shewale Jay Vinayak
# Project Number: 1
# Project Title: Differentiated Thyroid Cancer Recurrence using Decision Tree Decision Treebased Learning Mode

import pandas as pd 
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from DecisionTree import DecisionTree
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


df = pd.read_csv('Thyroid_Diff.csv')


X = df.drop('Recurred', axis=1)
Y = df['Recurred']

#########################################
#      USING ID3 IMPLEMENTATION
#########################################
print("\n\nUSING ID3 IMPLEMENTATION\n\n")

kf = KFold(n_splits=5, shuffle=True, random_state=4) 
sum=0
for i,(train_index, test_index) in enumerate(kf.split(X)):
    X_train, Y_train = X.iloc[train_index], Y.iloc[train_index] 
    X_test, Y_test = X.iloc[test_index], Y.iloc[test_index] 

    clf = DecisionTree(max_depth=5)
    print(f"\nTRAINING FOR FOLD {i}")
    clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_test)
    accuracy = clf.accuracy(y_pred, Y_test)

    print(f"fold {i} accuracy= {accuracy*100}") 
    
    print(y_pred.value_counts())
    print(f"Total number of nodes = {clf.count_total_nodes()}")

    print(classification_report(Y_test, y_pred))

    print(f"PRUNING ON FOLD {i}")
    clf.reduced_err_prunning(X_test, Y_test)
    accuracy = clf.accuracy(clf.predict(X_train), Y_train)
    print(f"Accurary on training data (after pruning) = {accuracy}")
    sum += accuracy

    # saving the plot of tree in ./plots directory
    try:
        clf.draw_tree(f"plots/fold_{i}.png")
    except:
        pass 
    

print(f"\n\nAverage accuracy of 5 folds: {sum*100/5}")



######################################### 
#       USING SKLEARN LIBRARY 
#########################################
    
print("\n\nUSING SKLEARN LIBRARY\n\n")
    
df_en=df.copy()

def label_encode_column(column):
    unique_values = np.unique(column)
    encoding_map = {value: index for index, value in enumerate(unique_values)}
    return column.map(encoding_map)

# Apply label encoding to all object columns
object_columns = df.select_dtypes(include=['object']).columns
for col in object_columns:
    df_en[col] = label_encode_column(df[col])

X = df.drop('Recurred', axis=1)
Y = df['Recurred']
# X.head()

X=np.array(X)
Y=np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
df_en.to_csv('output.csv')

with open('output.csv', 'r') as read_obj:
	csv_reader = csv.reader(read_obj)
	str_data = list(csv_reader)[1:]
	dataset = []
	for i in range(len(str_data)):
		try:
			temp = list(map(int, str_data[i][1:17]))
			temp.append(int(str_data[i][17]))
			dataset.append(temp)
		except:
			print(str_data[i])
               

# Separate labels and features for each data item
# print(dataset[:5])
X, y = [row[:16] for row in dataset], [row[16] for row in dataset]
# print(X)
# print(y)

# create 5 fold cross validation set (n_splits=5)
rn = range(1, len(dataset))
kf5 = KFold(n_splits=5, shuffle=True, random_state=4)

# traverse over each split
x=1
for train_index, test_index in kf5.split(X):
    print(f"\nTRAINING FOR FOLD {x}\n")
	# print(train_index, test_index)
    X_train, X_test = np.array([X[i] for i in train_index]), np.array([X[i] for i in test_index])
    y_train, y_test = np.array([y[i] for i in train_index]), np.array([y[i] for i in test_index])
	# X_train, X_test = [train_index], X[test_index]
	# y_train, y_test = y[train_index], y[test_index]
	
	# Save the datasets to file to process it from file separately
	
	# create decision tree using the sklearn library
    clf = DecisionTreeClassifier()
	# more at: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['No', 'Yes']))
    x=x+1

    ### PRUNING OF MAX_LEVEL ###

X = df_en.drop('Recurred', axis=1)
Y = df_en['Recurred']
X=np.array(X)
Y=np.array(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def prune_tree_and_evaluate(X_train, y_train, X_val, y_val, max_depth_range):
    train_accuracies = []
    val_accuracies = []

    for max_depth in max_depth_range:
        # Train decision tree classifier
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(X_train, y_train)

        # Evaluate on training set
        train_preds = clf.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_preds)
        train_accuracies.append(train_accuracy)

        # Evaluate on validation set
        val_preds = clf.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_preds)
        val_accuracies.append(val_accuracy)

        # Prune tree using Reduced Error Pruning
        # (You need to implement the pruning logic)

    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.plot(max_depth_range, train_accuracies, label='Training Accuracy')
    plt.plot(max_depth_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs. Max Depth')
    plt.legend()
    plt.show()

max_depth_range = range(1, 15)

# Run pruning and evaluation
prune_tree_and_evaluate(X_train, y_train, X_test, y_test, max_depth_range)
