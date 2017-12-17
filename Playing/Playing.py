

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
import math



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

headless = False
def main():
    global headless
    if (raw_input("Do you want to run the entire skript headless? (y/n)") == "y"):
        headless = True
    dataset = loadData()
    generateOverview(dataset)

    #Now that we got an overview about the data, let us start preparing for machine learning.
    #Machine learning algorithms do not like categorical values (such as labels). Multiple encoding mechanisms exist. we will use two of them.
    #One Hot Encoding:
    dataset_OneHot = dataset.copy()
    del dataset_OneHot['class']
    dataset_OneHot = pandas.get_dummies(dataset_OneHot)
    dataset_OneHot['class'] = dataset['class']
    if len(dataset['class'].unique()) == 2:
        dataset_OneHot['class'] = sklearn.preprocessing.LabelEncoder().fit_transform(dataset_OneHot['class'])


    #A linear encoding of just assigning numerical values for each category:
    dataset_Lbl = dataset.copy()
    encoders = {}
    for column in dataset_Lbl.columns:
        if dataset_Lbl.dtypes[column] == numpy.object:
            encoders[column] = sklearn.preprocessing.LabelEncoder()
            dataset_Lbl[column] = encoders[column].fit_transform(dataset_Lbl[column])

    heatmaps(dataset_OneHot,'OneHot')
    heatmaps(dataset_Lbl,'Label')


    #further data preparation:
    targetsOH = dataset_OneHot['class']
    targetsLBL = dataset_Lbl['class']
    del dataset_OneHot['class']
    del dataset_Lbl['class']

    x_train_OH, x_test_OH, y_train_OH, y_test_OH = sklearn.model_selection.train_test_split(dataset_OneHot, targetsOH, test_size = 0.4, random_state=0)

    x_train_Lbl, x_test_Lbl, y_train_Lbl, y_test_Lbl = sklearn.model_selection.train_test_split(dataset_Lbl, targetsLBL, test_size = 0.4, random_state=0)

    dataset_OneHot['class'] = targetsOH
    dataset_Lbl['class'] = targetsLBL
    
    algorithmTrialDefault(dataset_OneHot,'OneHot',x_train_OH,y_train_OH)
    algorithmTrialDefault(dataset_Lbl,'Label',x_train_Lbl,y_train_Lbl)
    gs_LR_OH = logisticRegressionParameterTrial(dataset_OneHot,'OneHot',x_train_OH,y_train_OH)
    gs_LR_LBL = logisticRegressionParameterTrial(dataset_Lbl,'Label',x_train_Lbl,y_train_Lbl)
    gs_KNN_OH = knnParameterTrial(dataset_OneHot,'OneHot',x_train_OH,y_train_OH)
    gs_KNN_LBL = knnParameterTrial(dataset_Lbl,'Label',x_train_Lbl,y_train_Lbl)


    import code
    code.interact(local=locals())

def loadData():
    #This code loads the dataset itself and defines the headers.
    print('Downloading Data...')
    #----Wine Dataset----
    #headerNames = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash  ', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280_OD315 of diluted wines', 'Proline']
    #dataset = pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None, sep=',', names=headerNames)
    #--------------------
    #-----Wine Quality dataset------
    dataset = pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', header=0, sep=';')
    dataset['class'] = dataset['quality']
    del dataset['quality']
    headerNames = dataset.columns
    #--------------------------------
    #----Car evaluation Dataset----
    #headerNames = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    #dataset = pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', header=None, sep=',', names=headerNames)
    #----Car evaluation Dataset----

    print("Download complete.")
    print('------------------------------------------')
    print
    #print(dataset)
    return dataset
def generateOverview(dataset):
    global headless
    #It usually is a good start to get a first idea of some statistics in the data.
    #In order to do so, we will print charts which show the distribution of the different values for all the features.
    print('------------------------------------------')
    headerNames = dataset.columns
    if ((headless) or raw_input("Do you want to see the data Overview? (y/n)") == "y"):
        print('Generating Data overview.')
        
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
                        #print(currentHeader)
                        #print(headerNames[currentHeader])
                    
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
        print('Data Overview completed. Please see overview.png.')
        if headless:
            plt.close()
        else:
            plt.show()
    print('------------------------------------------')
    print
    
    #Since the above code creates an overview, there should also be an option to export the figures one by one.
    #If some of the figures before have been too small, this should create a good looking figure for each feature
    print('------------------------------------------')
    if ((headless) or raw_input("Do you want to see the data Overview as single figures? (y/n)") == "y"):
        print ('Generating statistics for individual features.')
        for i in range(0,len(headerNames)):
            fig = plt.figure(i+1)
            if dataset.dtypes[headerNames[i]] == numpy.object:
                dataset.groupby(headerNames[i]).size().plot(kind='bar')
            else:
                dataset[headerNames[i]].hist()
                plt.xticks(rotation="vertical")
            #dataset.groupby(headerNames[currentHeader]).size().plot(kind='bar')
            #print(i)
            #print(headerNames[i])
            plt.subplots_adjust(left=0.08, top=0.95, bottom = 0.35, right = 0.95)
            plt.xlabel('')
            plt.title(headerNames[i])
            plt.xticks(rotation='vertical')
            plt.savefig('%s.png' % (headerNames[i]))
            print ('Statistic for Feature: %s exorted to file %s.png' % (headerNames[i], headerNames[i]))
        print ('Statsistic generation done. Please see the mentioned individual files for each feature.')
        if headless:
            plt.close('all')
        else:
            plt.show()    
    print('------------------------------------------')
    print               
def heatmaps(dataset, encoding):
    global headless
    print('------------------------------------------')
    if ((headless) or raw_input("Do you want to see the heatmaps of your direct data correlations? (y/n)") == "y"):
        print('Generating the Heatmap of %s encoded data' % (encoding))
        fig = plt.figure(1,figsize=(20,20))
        sns.heatmap(dataset.corr(), square=True, center=0, linewidth=0.5, cmap='seismic')
        plt.title('Heatmap of %s Encoded Data' % (encoding))
        plt.savefig('Heatmap_%s.png' % (encoding))
        print('Heatmap of %s encoded data done, please see file Heatmap_%s.png' % (encoding,encoding))
        if headless:
            plt.close()
        else:
            plt.show()
    print('------------------------------------------')
    print
def algorithmTrialDefault(dataset, encoding, x_train, y_train):   
    global headless
    print('------------------------------------------')
    if ((headless) or raw_input("Do you want to perform training based on %s Encoded data? (y/n)" % (encoding)) == "y"):
        print('Trying different algorithms:')
        #Test options and evaluation metric
    
        #If the dataset contains only binary data in the class, using the F1 scoring system might make more sense.
        #For comparison, we will use both the accuracy score and the F1 system.
        if dataset['class'].isin([0,1]).all():
            scoringMetric = ['accuracy','f1']
            refitMetric = 'f1'
        else:
            scoringMetric = 'accuracy'
            refitMetric = 'accuracy'
        # Spot Check Algorithms
        seed = 7
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier(n_jobs=-1)))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC(max_iter=20)))
        # evaluate each model in turn
        results = []
        names = []
        print("----------------------------")
        print("Starting training...")
        print("Algorithm comparison (based on %s Encoding):" % (encoding))
        print("Algorithm:   ScoringModel: Mean (std)")
        for name, model in models:
            msg = "%s:  " % (name)
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_validate(model, x_train, y_train, cv=kfold, scoring=scoringMetric)
            for scoringResult in (scoringResult for scoringResult in cv_results if scoringResult.find('test') != -1):
                msg = "%s %s: %f (%f)" % (msg,scoringResult,cv_results[scoringResult].mean(), cv_results[scoringResult].std() )
                #print cv_results[scoringResult]
            results.append(cv_results)
            names.append(name)
            #msg = "%s:  %f  (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
        print('Training done')
    print('------------------------------------------')
    print
def logisticRegressionParameterTrial(dataset, encoding, x_train, y_train):
    global headless
    print('------------------------------------------')

    if ((headless) or raw_input("Do you want to test different parameters for Logistic regression (%s Encoded Data)? (y/n)" % (encoding)) == "y"):
        print('Starting Logistic Regression Parameter Trial based on %s encoded Data. This might take a while' % (encoding))
        if dataset['class'].isin([0,1]).all():
            scoringMetric = ['accuracy','f1']
            refitMetric = 'f1'
        else:
            scoringMetric = 'accuracy'
            refitMetric = 'accuracy'
        seed=7
        LR = LogisticRegression()
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        parametersgrid = [
        {'C': range(1,11), 'penalty': ['l2'], 'solver': ['sag'], 'n_jobs': [-1]}, 
        {'C': range(1,11), 'penalty': ['l1','l2'], 'solver': ['liblinear', 'saga'], 'n_jobs': [-1]} 
        ]
        print ('The following Parameter grid will be used:')
        print parametersgrid
        gs = model_selection.GridSearchCV(LogisticRegression(), n_jobs=1, param_grid=parametersgrid, scoring=scoringMetric, cv=kfold, refit=refitMetric, verbose=1)
        gs.fit(x_train, y_train)
        print('Training done, these are the best parameters:')
        print ('Score: %f Parameters: %r' % (gs.best_score_, gs.best_params_))
        print('------------------------------------------')
        print
        return gs
    print('------------------------------------------')
    print
def knnParameterTrial(dataset, encoding, x_train, y_train):
    global headless
    print('------------------------------------------')
    if ((headless) or raw_input("Do you want to test different parameters for KNN classification (%s Encoded Data)? (y/n)" % (encoding)) == "y"):
        print('Starting K-neares Neighbour Parameter Trial based on %s encoded Data. This might take a while' % (encoding))
        if dataset['class'].isin([0,1]).all():
            scoringMetric = ['accuracy','f1']
            refitMetric = 'f1'
        else:
            scoringMetric = 'accuracy'
            refitMetric = 'accuracy'
        seed=7
        LDA = KNeighborsClassifier()
        kfold = model_selection.KFold(n_splits=5, random_state=seed)
        parametersgrid = [
        {'n_neighbors': range(1,len(dataset.columns), (len(dataset.columns) / 10) ), 'weights': ['uniform', 'distance'], 'n_jobs': [-1]} 
        ]
        print ('The following Parameter grid will be used:')
        print parametersgrid
        gs = model_selection.GridSearchCV(KNeighborsClassifier(), n_jobs=1, param_grid=parametersgrid, scoring=scoringMetric, cv=kfold, refit=refitMetric, verbose=1)
        gs.fit(x_train, y_train)
        print('Training done, these are the best parameters:')
        print ('Score: %f %r' % (gs.best_score_, gs.best_params_))
        print('------------------------------------------')
        print
        return gs
    print('------------------------------------------')
    print


if __name__ == "__main__": main()
