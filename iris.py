'''
How much data:
Number of features: 2
Target: Predict the species of a new iris
'''

'''
Since we have measurements for which we know the correct species of iris, this is a supervised learning
problem. In this problem, we want to predict one of several options (the species of irs). This is an 
example of a classification problem. The possible outputs are called classes.
Since every iris in the dataset belongs to one of three classes this proble is a three class
classification problem.
'''

# Using iris dataset
from sklearn.datasets import load_iris

# Load iris dataset
# The iris object that is retured by load_iris is a Bunch object, which is very similar
# to a dictionary. It contains keys and values.
# dict_keys(['DESCR', 'data', 'target_names', 'feature_names', 'target'])
# The value to the key DESCR is a short description of the dataset. We show the beginning of the
# description herer. Feel free to look up the rest yourself.
iris = load_iris()

# Description
print(iris['DESCR'][:193] + "\n...")

# The value with key target_names is an array of strings, containing the species of flower
# that we want to predict:
print(iris['target_names'])
print(iris['target_names'].dtype)
print(type(iris['target_names']))

print("Feature names")
print(iris['feature_names'])

print("Data")
print(iris['data'])

print("Data dimenstion")
print(iris['data'].shape)

# some of iris data
print(iris['data'][:5])

# The target arrary contains the species of each of the flowers that were measured, aso
# as a numpy array.
# The target is a one-dimensional array, with one entry per flower:
print(iris['target'].shape)

# targets
print(iris['target'])

# Scikit-learn contains a function that shuffles the dataset and splits it for you, the train_test_split function
# This function extracts 75% of the rows in the data as the training set, together with the corresponding
# labels for this data. The remaining 25% of the data, together with the remaining labels are declared 
# as the test set.
# In Scikit-learn, data is usually denoted with a capital X, while labels are denoted by a lower-case y.

from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test = train_test_split(iris['data'], iris['target'], random_state = 0)

'''
The train_test_split function shuffles the dataset using a pseudo random number generator before making the
split. If we would take the last 25% of the data as a test set, all the data point would have the label 2
as the data points are sorted by the label. Using a tests set containing only one of the three classes 
would not tell us much about how well we generalize, so we shuffle our data, to make sure the test data 
contains data from all classes.
'''

# shape of train data
print(X_train.shape)

# shape of test data
print(X_test.shape)

'''
Before building a machine learning model, it is often a good idea to inspect the data, to see
if the task is easily solvable without machine learning, or if the desired information might not 
be contained in the data
Inspecting your data is a good way to find abnomalities and penculiarities 
One of the best way to inspect data is to visualize it. One way to do this is by using a scatter plot.
A scatter plot of the data puts one feature along the x-axis, one feature along with the y-axis , and draws 
a dot for each data point.
Unfortunately, computer screens have only two dimensions, which allows us to only plot two (maybe three) 
features at a time. It is difficult to plot datasets with more than three features this way.
One way around this problem is to do a pair plot, which looks at all pairs of two features. If you have 
a small number of features, such as the four we have here, this is quite reasonable. You should keep 
in mind that a pair plot does not show the interaction of all of features at once, so some interesting 
aspects of the data may not be revealed when visualizing it this way.
'''
import matplotlib.pyplot as plt 
import numpy as np

# Create subplots 
# 3 x 3 plots
fig, ax = plt.subplots(3, 3, figsize=(15, 15))

# processing
for i in range(3):
	for j in range(3):
		ax[i, j].scatter(X_train[:, j], X_train[:, i + 1], c=y_train, s=60)
		ax[i, j].set_xticks(())
		ax[i, j].set_yticks(())
		if i == 2:
			ax[i, j].set_xlabel(iris['feature_names'][j])
		if j == 0:
			ax[i, j].set_ylabel(iris['feature_names'][i + 1])
		if j > i:
			ax[i, j].set_visible(False)
# title
plt.suptitle("iris_pairplot")

plt.show()





