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