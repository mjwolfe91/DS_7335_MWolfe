import numpy as np
from warnings import simplefilter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from itertools import product
import statistics
import os

simplefilter(action='ignore', category=FutureWarning)
output_dir = 'output' + '/'
output_dir_best = output_dir + 'BestResult'
output_dir_GridSearchResults = output_dir + '/' + 'CompleteResults'

class DeathToGridSearch:

    def __init__(self, clfDict, clfList, data):
        self.clfDict = clfDict
        self.clfList = clfList
        self.data = data

    def run_KFold(self, a_clf, clf_params,
                  resultsDict = {}):

        X, y, n_folds = self.data
        kf = KFold(n_splits=n_folds)

        for ids, (train_index, test_index) in enumerate(kf.split(X, y)):
            clf = a_clf(**clf_params)

            clf.fit(X[train_index], y[train_index])

            pred = clf.predict(X[test_index])

            resultsDict[ids] = {'classifier': clf,
                                'train_index': train_index,
                                'test_index': test_index,
                                'accuracy': accuracy_score(y[test_index], pred)}

        return resultsDict

    def run_CV(self, classifier, clf_hyperparameters, cvResultsDict = {}):

        X, y, n_folds = self.data
        sKFold_CV = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=0)

        for k, (train_index, test_index) in enumerate(sKFold_CV.split(X=X, y=y)):

            clf = classifier(**clf_hyperparameters)
            clf.fit(X[train_index], y[train_index])
            pred = clf.predict(X[test_index])
            if classifier == LogisticRegression:
                featureWeights = clf.coef_
            elif classifier == RandomForestClassifier:
                featureWeights = clf.feature_importances_
            elif classifier == DecisionTreeClassifier:
                featureWeights = clf.feature_importances_

            cvResultsDict[k] = {'fold': k + 1,
                                'classifier': clf,
                                'train_index': train_index,
                                'test_index': test_index,
                                'accuracy': accuracy_score(y[test_index], pred),
                                'featureWeights': featureWeights}
        return cvResultsDict

    def run_gridsearch_classifiers(self, clfAccuracyDict = {}, clfFeatureImportanceDict = {}):

        for clf in self.clfList:
            for k1, v1 in self.clfDict.items():
                if k1 in str(clf):
                    k2, v2 = zip(*v1.items())
                    for values in product(*v2):
                        hyperSet = dict(zip(k2, values))
                        results = self.run_CV(clf, hyperSet)
                        for key in results:
                            k1 = results[key]['classifier']
                            v1 = results[key]['accuracy']
                            v2 = results[key]['featureWeights']
                            k1Test = str(k1)
                            k1Test = k1Test.replace('            ', ' ')
                            k1Test = k1Test.replace('          ', ' ')
                            if k1Test in clfAccuracyDict:
                                clfAccuracyDict[k1Test].append(v1)
                            else:
                                clfAccuracyDict[k1Test] = [v1]

                            if k1Test in clfFeatureImportanceDict:
                                clfFeatureImportanceDict[k1Test].append(v2)
                            else:
                                clfFeatureImportanceDict[k1Test] = [v2]

        return clfAccuracyDict, clfFeatureImportanceDict

    def run_gridsearch(self, clfAccuracyDict={}):

        for clf in self.clfList:
            for k1, v1 in self.clfDict.items():
                if k1 in str(clf):
                    k2, v2 = zip(*v1.items())
                    for values in product(*v2):
                        hyperSet = dict(zip(k2, values))
                        results = self.run_KFold(clf, hyperSet)
                        for key in results:
                            k1 = results[key]['classifier']
                            v1 = results[key]['accuracy']
                            k1Test = str(k1)
                            k1Test = k1Test.replace('            ', ' ')
                            k1Test = k1Test.replace('          ', ' ')
                            if k1Test in clfAccuracyDict:
                                clfAccuracyDict[k1Test].append(v1)
                            else:
                                clfAccuracyDict[k1Test] = [v1]
        return clfAccuracyDict

    def get_best_score(self, gridsearch_results):

        filename = 'best_clf_Boxplot'
        best_accuracy = 0

        for classifier, accuracy in gridsearch_results.items():

            if best_accuracy < statistics.mean(accuracy):
                best_accuracy = statistics.mean(accuracy)
                optimal_classifier = classifier
                optimal_hyperparameters = optimal_classifier, best_accuracy

                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot(1, 1, 1)
                plt.boxplot(accuracy, vert=False)
                ax.set_title(str(optimal_classifier) + "\nAccuracy: " + str(best_accuracy), fontsize=30)
                ax.set_xlabel('Accuracy Scores', fontsize=25)
                ax.set_ylabel('Classifier Accuracy (By K-Fold)',     fontsize=25)
                ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1))
                ax.yaxis.set_ticks(np.arange(0, 1, 1))
                ax.xaxis.set_tick_params(labelsize=20)
                ax.yaxis.set_tick_params(labelsize=20)
                plt.savefig(output_dir + '/' + filename, bbox_inches='tight')
                plt.show()
                try:
                    plt.savefig(output_dir + '/' + filename, bbox_inches='tight')
                except IOError:
                    os.mkdir(output_dir)
                    plt.savefig(output_dir + '/' + filename, bbox_inches='tight')

        print("\n*Model with best accuracy: ", optimal_hyperparameters[1], "\nClassifier & Parameters: \n", optimal_hyperparameters[0], "\n*")

    def get_features_with_importance(self, gridsearch_results, gridsearch_feature_weights, columns):

        filename = 'best_clf_feature_importances'
        best_accuracy = 0

        for classifier, accuracy in gridsearch_results.items():

            if best_accuracy < statistics.mean(accuracy):
                best_accuracy = statistics.mean(accuracy)
                optimal_classifier = classifier
                feature_weights = gridsearch_feature_weights[optimal_classifier]

                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot(1, 1, 1)
                plt.bar(feature_weights, vert=False)
                ax.set_title(str(optimal_classifier) + " Feature Importances", fontsize=30)
                ax.set_xlabel('Features', fontsize=25)
                ax.set_ylabel('Feature Importance',     fontsize=25)
                ax.xaxis.set_ticks(columns)
                ax.yaxis.set_ticks(np.arange(0, 1, 1))
                ax.xaxis.set_tick_params(labelsize=20)
                ax.yaxis.set_tick_params(labelsize=20)
                plt.savefig(output_dir + '/' + filename, bbox_inches='tight')
                plt.show()
                try:
                    plt.savefig(output_dir + '/' + filename, bbox_inches='tight')
                except IOError:
                    os.mkdir(output_dir)
                    plt.savefig(output_dir + '/' + filename, bbox_inches='tight')
