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
from sklearn.neighbors import KneighborsClassifier

# Create a Knn instance
knn = KneighborsClassifier(n_neighbors = 1)