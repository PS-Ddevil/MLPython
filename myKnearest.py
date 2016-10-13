'''
Building your first model: k nearest neighbors
Now we can start building the actual machine learning model. There are many classification algorithms
in scikit-learn that we could use. Here we will use a k nearest neighbors classifior, which is easy to understand.
Building this model only consists of storing the training set. To make a prediction for a new data point
Then, it and assigns the label of this closest data training point to the new data point. 

The k in k nearest neighbors stands for the fact that instead of using only the closest neighbor 
to the new data point, we can consider any fixed number k of neighbors in the training (for example
the closest three or five neighbors). Then, we can make a prediction using the majority class among
these neighbors. 
Let's use  only a single neighbor for now

All machine learning models in scikit-learn are implemented in their own class, which are called 
Estimator classes. The k nearest neighbors classification algorithm is implemeted in the 
KNeighborsClassifier class in the neighbors module
Before we can use the module, we need to instantiate the class into an object. This is when we will
set any parameters of the module. The single parameter of the KneighborsClassifier is the number of
neighbors, which we will set to one:
'''

# using KneighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# load iris
iris = load_iris()

# create trainning set and test set
X_train, X_test , y_train, y_test = train_test_split(iris['data'], iris['target'], random_state = 0)

# Create a Knn instance
knn = KNeighborsClassifier(n_neighbors = 1)

'''
# Knn object encapsulates the algorithm to build the model from the trainning data, as well the 
# the algorithm to make predictions on the new data points.
It will also hold the information the algorithm has extracted from the trainning data.
In the case of KNeighborsClassifier, it will just store the trainning set.
To build the model on the trainning set, we call the fit method of the knn object,
which takes as arguments the numpy array X_train containing the trainning data and the numpy array
y_train of the corresponding trainning labels:
'''

# Fit model
knn.fit(X_train, y_train)

# Model parameters
KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski', 
	metric_params = None, n_jobs = 1, n_neighbors = 1, p = 2, weights = 'uniform')

# We can now make predictions using this model on new data, for which we might not know the
# correct labels
# Imagine we found an iris in the wild with a sepal length of 5cm, a sepal width of 2.9cm,
# a petal length of 1cm and a petal width of 0.2cm. What species of iris would this be??
# We can put this data into a numpy array, again with the shape number of samples (one) times
# number of features (four):
X_new = np.array([[5, 2.9, 1, 0.2]])
X_new1 = np.array([[6, 3.1, 3, 0.5], [3, 3.5, 4, 0.6]])
# shape of the new example
print(X_new.shape)
print(X_new1.shape)
# To make prediction we call the predict method of the knn object:
prediction = knn.predict(X_new)
prediction1 = knn.predict(X_new1)

# The class of the new sample
print(prediction)
print(prediction1)

# Name of the class
print(iris['target_names'][prediction])
print(iris['target_names'][prediction1])

'''
Our model predicts that this new iris belongs to the class 0, meaning its species is Setosa.]
But how do we know whether we can trust our model ? WE don't know the correct species of this sample
which is the whole point of building the model.
Evaluating the model. This is where the test set that we created earlier comes in.
This data was not used to build the model, but we do know that the correct species are fore each
irirs in the test set.
We can make a prediction for an iris in the test data, and compare it against its lable
(the known species). We can measure how well the model works by computing the accuracy, which is the 
fraction of flowers for which the right species was predicted:

'''

# make prediction on the test set
y_pred = knn.predict(X_test)

# The accuracy
print(np.mean(y_pred == y_test))

# We can also use the score method of the knn object, which will compute the test set accuracy 
# for us:
print(knn.score(X_test, y_test))
