import sys
from matplotlib.backends.backend_template import new_figure_manager
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
import seaborn as sns
import math

#This code loads the dataset itself and defines the headers.
headerNames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']
print('Downloading Data...')
dataset = pandas.read_csv('adult.data', header=None, sep=',', names=headerNames)

print("Download complete. The fnlwgt feature is not needed for our prediction. Let us delete it.")
del dataset['fnlwgt']
del headerNames[2]
print('---------------')
print(dataset)

#It usually is a good start to get a first idea of some statistics in the data.
#In order to do so, we will print charts which show the distribution of the different values for all the features.
if raw_input("Do you want to see the data Overview? (y/n)") == "y":
    numberOfCols = 4
    numberOfRows = int(math.ceil(len(headerNames) / float(numberOfCols) ))
    #fig = plt.figure(figsize=(10,10))
    #The figsize would have to be adjusted for other datasets with more features.
    fig, splts = plt.subplots(numberOfCols,numberOfRows, figsize=(15,18))
    for i in range(0,numberOfCols):
        for j in range(0,numberOfRows):
                currentHeader = i + j + (i*(numberOfCols-1))
                if (currentHeader < len(headerNames)):
                    #ax = fig.add_subplot(numberOfRows,numberOfCols,currentHeader+1)
                    print(currentHeader)
                    print(headerNames[currentHeader])
                    
                    #If we have values that are numerical, a histogram is a good option for plotting.
                    #For features that are categorical, a barchart is beneficial
                    if dataset.dtypes[headerNames[currentHeader]] == numpy.object:
                        dataset.groupby(headerNames[currentHeader]).size().plot(kind='bar', ax = splts[i,j])
                    else:
                        dataset[headerNames[currentHeader]].hist(ax = splts[i,j] )
                        plt.xticks(rotation="vertical")
                    #dataset.groupby(headerNames[currentHeader]).size().plot(kind='bar', ax = splts[i,j] )
                    splts[i,j].set_title(headerNames[currentHeader])
                    splts[i,j].set_xlabel('')
    #This makes the output a little bit nicer by adjusting the borders of the subplots
    plt.subplots_adjust(left=0.05, top=0.95, bottom = 0.2, right = 0.95, hspace=1, wspace=0.3)
    #plt.subplots_adjust(left=0.05, top=0.95, bottom = 0.30, right = 0.95)
    #plt.subplots_adjust(hspace=0.7, wspace=0.3)
    
    #Save it as a png file and then display it to the user. Note: plt.show() also causes the destruction routing for the plot. savefig() does not.
    plt.savefig('overview.png')
    plt.show()
    
#Since the above code creates an overview, there should also be an option to export the figures one by one.
#If some of the figures before have been too small, this should create a good looking figure for each feature
if raw_input("Do you want to see the data Overview as single figures? (y/n)") == "y":
    for i in range(0,len(headerNames)):
        fig = plt.figure(i+1)
        if dataset.dtypes[headerNames[i]] == numpy.object:
            dataset.groupby(headerNames[i]).size().plot(kind='bar')
        else:
            dataset[headerNames[i]].hist()
            plt.xticks(rotation="vertical")
        #dataset.groupby(headerNames[currentHeader]).size().plot(kind='bar')
        print(i)
        print(headerNames[i])
        plt.subplots_adjust(left=0.08, top=0.95, bottom = 0.35, right = 0.95)
        plt.xlabel('')
        plt.title(headerNames[i])
        plt.xticks(rotation='vertical')
        plt.savefig(headerNames[i])
    plt.show()                    


#Now that we got an overview about the data, let uss startn preparing for machine learning.
#Machine learning algorithms do not like categorical values (such as labels). Multiple encoding mechanisms exist. we will use two of them.
#One Hot Encoding:
dataset_OneHot = dataset.copy()
del dataset_OneHot['class']
dataset_OneHot = pandas.get_dummies(dataset_OneHot)
dataset_OneHot['class'] = dataset['class']

#A linear encoding of just assigning numerical values for each category:
dataset_Lbl = dataset.copy()
encoders = {}
for column in dataset_Lbl.columns:
    if dataset_Lbl.dtypes[column] == numpy.object:
        encoders[column] = sklearn.preprocessing.LabelEncoder()
        dataset_Lbl[column] = encoders[column].fit_transform(dataset_Lbl[column])

if raw_input("Do you want to see the heatmaps of your direct data correlations? (y/n)") == "y":
    fig = plt.figure(1,figsize=(20,20))
    sns.heatmap(dataset_OneHot.corr(), square=True, center=0, linewidth=0.5, cmap='seismic')
    plt.title('Heatmap of One Hot Encoded Data')
    plt.savefig('Heatmap_OneHot.png')
    fig = plt.figure(2,figsize=(20,20))
    sns.heatmap(dataset_Lbl.corr(), square=True, center=0, linewidth=0.5, cmap='seismic')
    plt.title('Heatmap of Label Encoded Data')
    plt.savefig('Heatmap_Label.png')
    plt.show()

#Now we are prepared for performing some machine learning:
targetsOH = dataset_OneHot['class']
targetsLBL = dataset_Lbl['class']
del dataset_OneHot['class']
del dataset_Lbl['class']

x_train_OH, x_test_OH, y_train_OH, y_test_OH = sklearn.model_selection.train_test_split(dataset_OneHot, targetsOH, test_size = 0.4, random_state=0)

x_train_Lbl, x_test_Lbl, y_train_Lbl, y_test_Lbl = sklearn.model_selection.train_test_split(dataset_Lbl, targetsLBL, test_size = 0.4, random_state=0)

dataset_OneHot['class'] = targetsOH
dataset_Lbl['class'] = targetsLBL

#Test options and evaluation metric
seed = 7
#If the dataset contains only binary data in the class, using the F1 scoring system might make more sense.
#For comparison, we will use both the accuracy score and the F1 system.
if dataset['class'].isin([0,1]).all():
    scoring = ['accuracy','f1']
else:
    scoring = 'accuracy'
print("----------------------------")
print("Starting training...")
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(n_jobs=-1)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_jobs=-1)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(max_iter=20)))
# evaluate each model in turn
results = []
names = []
if raw_input("Do you want to perform training based on OneHot Encoded data? (y/n)") == "y":
    if dataset_OneHot['class'].isin([0,1]).all():
        scoring = ['accuracy','f1']
    else:
        scoring = 'accuracy'
    print("Algorithm comparison (based on One Hot Encoding):")
    print("Algorithm:   ScoringModel: Mean (std)")
    for name, model in models:
        msg = "%s:  " % (name)
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_validate(model, x_train_OH, y_train_OH, cv=kfold, scoring=scoring)
        for scoringResult in (scoringResult for scoringResult in cv_results if scoringResult.find('test') != -1):
            msg = "%s %s: %f (%f)" % (msg,scoringResult,cv_results[scoringResult].mean(), cv_results[scoringResult].std() )
            #print cv_results[scoringResult]
        results.append(cv_results)
        names.append(name)
        #msg = "%s:  %f  (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    print('---------------------------------------')

if raw_input("Do you want to perform training based on Label Encoded data? (y/n)") == "y":
    print("Algorithm comparison (based on Label Encoding):")
    print("Algorithm:   ScoringModel: Mean (std)")
    if dataset_Lbl['class'].isin([0,1]).all():
        scoring = ['accuracy','f1']
    else:
        scoring = 'accuracy'
    for name, model in models:
        msg = "%s:  " % (name)
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_validate(model, x_train_Lbl, y_train_Lbl, cv=kfold, scoring=scoring)
        for scoringResult in (scoringResult for scoringResult in cv_results if scoringResult.find('test') != -1):
            msg = "%s %s: %f (%f)" % (msg,scoringResult,cv_results[scoringResult].mean(), cv_results[scoringResult].std() )
        results.append(cv_results)
        names.append(name)
        #msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    print('---------------------------------------')

if raw_input("Do you want to test different parameters for Logistic regression? (y/n)") == "y":
    LR = LogisticRegression()
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    parametersgrid = [
    {'C': range(0,11), 'penalty': ['l2'], 'solver': ['newton-cg', 'lbfgs', 'sag']}, 
    {'C': range(0,11), 'penalty': ['l1','l2'], 'solver': ['liblinear', 'saga']} 
    ]
    #gs = model_selection.GridSearchCV(LogisticRegression(), param_grid=parametersgrid, scoring=scoring, cv=kfold, refit='f1')
    #gs.fit(x_train_OH, y_train_OH)


#I want to be able to play around with the data manually during testing. The following lines will switch python to an interactive environment (exit with "quit()")
import code
code.interact(local=locals())
