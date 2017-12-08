import pandas


headerNames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']
print('Downloading Data...')
dataset = pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None, sep=',', names=headerNames)
print('---------------')
print(dataset)



#I want to be able to play around with the data manually during testing. The following lines will switch python to an interactive environment (exit with "quit()")
import code
code.interact(local=locals())
