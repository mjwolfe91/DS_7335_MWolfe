# Homework 1

import numpy as np
from warnings import simplefilter
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

simplefilter(action='ignore', category=FutureWarning)
# adapt this code below to run your analysis
# Recommend to be done before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each

def gridSearchBasic(**kwargs):
    for arg in kwargs.values():
        clf = LogisticRegression(random_state=0).fit(arg)
        return clf

# Recommend to be done before live class 3
# 2. expand to include larger number of classifiers and hyperparameter settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
# Recommend to be done before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
L = np.ones(M.shape[0])
n_folds = 5

clfDict = {'RandomForestClassifier':{
    'min_samples_split': [2,3,4],
    'n_jobs': [1,2]},
    'LogisticRegression':{'tol': [0.001,0.01,0.1]},
    'KNNeighborsClassifier': {
        'n_neighbors': [2,3,5,10,25],
        'algorithm': ['auto','ball_tree','brute'],
        'p': [1,2]
}}

data = (M, L, n_folds)

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

results = run(RandomForestClassifier, data, clf_hyper={})

#Example solution

clfsList = {RandomForestClassifier,LogisticRegression}
for clfs in clfsList:
    results = run(clfs, data, clf_hyper=clfDict)
    print(results)

#Example solution 2
def myClfHypers(clfsList):

    for clf in clfsList:
        clfString = str(clf)
        print('clf: ', clfString)

        for k1, v1 in clfDict.items():
            if k1 in clfString:
                for k2, v2 in v1.items():
                    print(k2)
                    for vals in v2:
                        print(vals)

myClfHypers(clfsList)

