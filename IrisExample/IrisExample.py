#following the Tutorial on:https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#First of all, we have to catch the data. For this, we will be using Pandas:
#For other datasets, both the URL and the names would have to get changed according to the dataset description.

print("Downloading data...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
print("Download complete.")
#Next, we will plot a couple of statistics of our data. This is pretty handy in order to manually check the data and to get an idea of it.

print("----------------------------")
print("Let´s take a look at the data itself:")
# shape
print("Shape:")
print(dataset.shape)

#Data itself. Just printing it will print the first and the last 30 entries.
print("----------------------------")
print("The data itself:")
print(dataset)

# descriptions
print("----------------------------")
print("a few statistics on the data:")
print(dataset.describe())

# class distribution
print("----------------------------")
print("Class distribution:")
print(dataset.groupby('class').size())

# box and whisker plots
print("----------------------------")
print("A few plots on the data:")
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
print("----------------------------")
print("Histograms:")
dataset.hist()
plt.show()

# scatter plot matrix
print("----------------------------")
print("scatterplot matrix:")
scatter_matrix(dataset)
plt.show()

# Split-out validation dataset
print("----------------------------")
print("Starting data processing...")
print("splitting training sets:")
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

print(results)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

