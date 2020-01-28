# Homework 1

import numpy as np
from warnings import simplefilter
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
from itertools import product
import csv

simplefilter(action='ignore', category=FutureWarning)

# adapt this code below to run your analysis
# Recommend to be done before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each

# Recommend to be done before live class 3
# 2. expand to include larger number of classifiers and hyperparameter settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings

# Recommend to be done before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

#M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
#L = np.ones(M.shape[0])

data = load_breast_cancer()
M, L = data.data, data.target
n_folds = 5

clfDict = {'RandomForestClassifier':{
    'min_samples_split': [2,3,5,6],
    'n_jobs': [3,5]},
    'LogisticRegression':{'tol': [0.001,0.0011,.005,.0055]},
    'KNeighborsClassifier': {
        'n_neighbors': [20,21,22,23,25,30],
        'algorithm': ['auto','ball_tree','brute'],
        'p': [3,4,5]},
    'QuadraticDiscriminantAnalysis':{'tol':[0.001,0.0011,.005,.0055],
    'store_covariance':[False,True]}
          }

data = (M, L, n_folds)

clfsList = {RandomForestClassifier,LogisticRegression,KNeighborsClassifier,QuadraticDiscriminantAnalysis}
clfAccuracyDict = {}

def run(a_clf, data, clf_hyper=clfDict):
  M, L, n_folds = data # unpack data container
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {} # classic explication of results

  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    clf = a_clf(**clf_hyper) # unpack parameters into clf is they exist

    clf.fit(M[train_index], L[train_index])

    pred = clf.predict(M[test_index])

    ret[ids]= {'clf': clf,
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}
  return ret

for clf in clfsList:
    for k1, v1 in clfDict.items():
        if k1 in str(clf):
            k2,v2 = zip(*v1.items())
            for values in product(*v2):
                hyperSet = dict(zip(k2, values))
                results = run(clf, data, hyperSet)
                for key in results:
                    k1 = results[key]['clf']
                    v1 = results[key]['accuracy']
                    k1Test = str(k1)
                    k1Test = k1Test.replace('            ',' ')
                    k1Test = k1Test.replace('          ',' ')
                    if k1Test in clfAccuracyDict:
                        clfAccuracyDict[k1Test].append(v1)
                    else:
                        clfAccuracyDict[k1Test] = [v1]

n = max(len(v1) for k1, v1 in clfAccuracyDict.items())

output_path = 'Homework1/output/'
filename_prefix = 'clf_Boxplots_'
csv_file = 'Data.csv'
try:
    with open(output_path + filename_prefix + csv_file, 'w', newline="") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in clfAccuracyDict.items():
            writer.writerow([key, value])
except IOError:
    print("I/O error")

plot_num = 1
left = 0.125
right = 0.9
bottom = 0.1
top = 0.6
wspace = 0.2
hspace = 0.2
for k1, v1 in clfAccuracyDict.items():
    fig = plt.figure(figsize =(10,10))
    ax = fig.add_subplot(1, 1, 1)
    plt.boxplot(v1)
    ax.set_title(k1, fontsize=25)
    ax.set_xlabel('Classifer Accuracy (By K-Fold)', fontsize=25)
    ax.set_ylabel('Frequency', fontsize=25)
    ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.set_ticks(np.arange(0, n + 1, 1))
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    plot_num_str = str(plot_num)
    filename = filename_prefix + plot_num_str
    plt.savefig(output_path + 'plots/' + filename, bbox_inches='tight')
    plot_num = plot_num + 1
plt.show()
