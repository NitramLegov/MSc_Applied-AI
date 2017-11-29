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
if raw_input("Do you want to see the boxplot? (y/n)") == "y":
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()


# histograms
print("----------------------------")
print("Histograms:")

if raw_input("Do you want to see the Histograms? (y/n)") == "y":
    dataset.hist()
    plt.show()

# scatter plot matrix
print("----------------------------")
print("scatterplot matrix:")

if raw_input("Do you want to see the scatter plot? (y/n)") == "y":
    scatter_matrix(dataset)
    plt.show()

# Split-out validation dataset
print("----------------------------")
print("Starting data processing...")
print("splitting training sets...")
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
print("----------------------------")
print("Starting training...")
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
print("Algorithm comparison:")
print("Algorithm: Mean (std)")
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

#print(results)

# Compare Algorithms
if raw_input("Do you want to see the boxplot for algorithm comparison? (y/n)") == "y":
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')    
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

# Make predictions on validation dataset
print("----------------------------")
print("Predicting values based on KNN:")
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)

print("Accuracy Score:")
print(accuracy_score(Y_validation, predictions))
print("----------------------------")
print("Confusion Matrix:")
print(confusion_matrix(Y_validation, predictions))
print("----------------------------")
print("Classification report:")
print(classification_report(Y_validation, predictions))

print("----------------------------")
print("Let us try different values for K")
print('K    ')
print('Accuracy')
knn_results=[]
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    knn_results.append(accuracy_score(Y_validation, predictions))
    print("%s   %f" % (k,knn_results[k-1]) )


if raw_input("Do you want to see the visual representation of the different results for K? (y/n)") == "y":
    plt.bar(range(1,21),knn_results,0.5,label='K-NN',tick_label=range(1,21))
    plt.title('K-NN Comparison')
    plt.show()
#print(knn_results)

#I want to be able to play around with the data manually during testing. The following lines will switch python to an interactive environment (exit with "quit()")
import code
code.interact(local=locals())