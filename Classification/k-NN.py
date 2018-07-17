import pandas as pd
import csv

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn import neighbors
from sklearn import datasets

iris_data = datasets.load_iris()
print ('Keys:', iris_data.keys())
print ('-' * 20)
print ('Data Shape:', iris_data.data.shape)
print ('-' * 20)
print ('Features:', iris_data.feature_names)
print ('-' * 20)
#
# #iris_data.loc[iris_data['class'] == 'versicolor', 'class'] = 'Iris-versicolor' #clean species labels
# #iris_data.loc[iris_data['class'] == 'Iris-setossa', 'class'] = 'Iris-setosa'
# X, y = iris_data.data, iris_data.target
#
# #define the model
# knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')
#
# #fit/train the new model
# knn.fit(X,y)
#
# #What species has a 2cm x 2cm sepal and a 4cm x 2cm petal?
# X_pred = [2, 2, 4, 2]
# output = knn.predict([X_pred,]) #use the model we just created to predict
#
# print ('Predicted Species:', iris_data.target_names[output])
# print ('Options:', iris_data.target_names)
# print ('Probabilities:', knn.predict_proba([X_pred, ]))
# print(output)

## load the iris data into a DataFrame
## Specifying column names.
iris_data = datasets.load_iris()
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
## map each iris species to a number with a dictionary and list comprehension.
iris_class = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
#iris_data['species_num'] = [iris_class[i] for i in iris_data.species]

## Create an 'X' matrix by dropping the irrelevant columns.
#X = iris_data.drop(['species', 'species_num'], axis=1)
#y = iris_data.species_num
X, y = iris_data.data, iris_data.target

## Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

## Instantiate the model with 5 neighbors.
knn = KNeighborsClassifier(n_neighbors=5)
## Fit the model on the training data.
knn.fit(X_train, y_train)
## See how the model performs on the test data.
print(knn.score(X_test, y_test))
print(knn.predict(X_test))
print("End of code")