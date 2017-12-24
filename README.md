# Applied-AI
This Repository contains code developed during the "Applied AI" course at SRH Heidelberg in Novermber / December 2017

# Prerequisites
This Project needs the following packages installed in your python environment:<br>
scipy<br>
numpy<br>
matplotlib<br>
pandas<br>
sklearn<br>
seaborn<br>

All these packages can be installed using pip.
# Purpose
The objective of this project is to analyze the Adult Dataset hosted by the UCI Machine learning repository.<br>
The main python script can be found in the folder AdultDataset. Running it will analyse the dataset.<br>
During the development of the mentioned script, effort was taken so that the script is as generic as possible and can be used with other datasets with minimal code adjustments.<br>

# Usage
If the code is supposed to be used with a different dataset, the following code adjustments be needed in the functions main and load_data.<br>
main:<br>
Comment out the following lines:<br>
dataset_flt = dataset[dataset["workclass"] != "?"]<br>
dataset_flt = dataset_flt[dataset_flt["occupation"] != "?"]<br>
dataset_flt = dataset_flt[dataset_flt["native-country"] != "?"]<br>
del dataset_flt['education']<br>

load_data:<br>
Adjust the following lines to whatever is needed to load your data into a DataFrame as delivered by pandas:
headerNames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']<br>
print('Downloading Data...')<br>
dataset = pandas.read_csv('adult.data', header=None, sep=', ', names=headerNames)<br>
print("Download complete. The fnlwgt feature is not needed for our prediction. Let us delete it.")<br>
del dataset['fnlwgt']<br>
del headerNames[2]<br>

Please note that it is important that the values to be predicted are in a column called "class".

If a dataset with non-binary classes is used, a slight adjustments also needs to be done in function algorithmTrialDefault:<br>
#results.append(cv_results)                                             should be uncommented<br>
results = pandas.DataFrame(results)                                     should be commented out<br>
print (results.sort_values(by='test_roc_auc', ascending=False))         should be commented out<br>

# Testing
This script was tested on python 2.7 and python 3.6<br>
It worked in both environments, but the performance was slightly better with 3.6<br>

# Running the Script
Each function of the script will prompt the user, whether it shall be executed or not.<br>
These prompts can be skipped by answering the prompt whether the skript shall be executed headless with "y".<br>
The following steps will be executed:
1. Loading the data into a DataFrame.
2. Generating a graphical overview of the data and export it into png files.
3. Generating heatmaps of feature correlations and export it into png files.
4. Trying different machine learning algorithms with their default parameters.
5. Executing a parameter optimization for the Logistic Regression Algorithm.
6. Executing a parameter optimization for the KNN Algorithm.
7. Executing a parameter optimization for the SVM Algorithm. Please expect a very long runtime (easily 10 hours) here.

An Analysis of the AdultDataset can be found in the file "Documentation.docx". This file describes my work done during the Applied AI course.
