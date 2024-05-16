from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()

np.random.seed(123)
test_idx = np.random.randint(0, len(iris.target), len(iris.target) // 3)

train_data = np.delete(iris.data, test_idx, axis=0)
train_target = np.delete(iris.target, test_idx)
test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best')
clf.fit(train_data, train_target)

print('\nThe target test data set is:\n', test_target)
print('\nThe predicted result is:\n', clf.predict(test_data))
print('\nAccuracy rate is:\n', accuracy_score(test_target, clf.predict(test_data)))

# Visualizing the tree using Matplotlib
fig, ax = plt.subplots(figsize=(12, 12))  # Set appropriate size according to your needs
tree.plot_tree(clf, filled=True, rounded=True,
               feature_names=iris.feature_names,
               class_names=list(iris.target_names))
plt.show()
