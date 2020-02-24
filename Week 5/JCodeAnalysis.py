import numpy as np
import numpy.lib.recfunctions as rfn
from collections import OrderedDict, Counter
from sklearn.metrics import accuracy_score, precision_score
import itertools as it
from itertools import product
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from collections import OrderedDict, Counter
import matplotlib.pyplot as plt
import statistics
import datetime
import json
import re
import os
import sys

from DeathToGridSearch.DeathToGridSearch import DeathToGridSearch

sys.path.append(r'C:\Users\micha\DS_7335_MWolfe\DeathToGridSearch')

def get_JCodes_with_claims(data):
    JcodeIndicies = np.core.defchararray.startswith(data['ProcedureCode'], prefix = 'J'.encode(), start=0, end=1)
    Jcodes = data[JcodeIndicies]
    print("There are " + str(len(Jcodes)) + " claim lines that have J-Codes.")
    return Jcodes

def get_provider_amounts(data):
    JCodeInNet = data[data['InOutOfNetwork'] == 'I'.encode()]
    print("$"+str(round(sum(JCodeInNet['ProviderPaymentAmount']),2))+" was paid for J-codes to providers for in-network claims.")

def provider_payments(data, n_top_providers, payments_dict = {}):
    Sorted_Jcodes = np.sort(data, order='ProviderPaymentAmount')[::-1]
    JCodePayments = rfn.merge_arrays([Sorted_Jcodes['ProcedureCode'], Sorted_Jcodes['ProviderPaymentAmount']], flatten = True, usemask = False)
    for JCode in JCodePayments:
        if JCode[0] in payments_dict.keys():
            payments_dict[JCode[0]] += JCode[1]
        elif JCode[0] not in payments_dict.keys():
            payments_dict[JCode[0]] = JCode[1]
    JCodes_AggregatedPayments = OrderedDict(sorted(payments_dict.items(), key=lambda JCode: JCode[1], reverse=True))
    top_providers = list(JCodes_AggregatedPayments.items())[0:n_top_providers]
    print("The top five J-codes based on payments to providers are: ")
    for val in top_providers:
        print("\t", val[0]," : $", round(val[1],2))


def paid_vs_unpaid(paid, unpaid):
    filename = 'PaidvsUnpaid_Claims.png'
    NumProviders_PaidJCode = len(paid)
    paid_AGG = [paid[k] for k in unpaid.keys()]
    unpaid_AGG = [unpaid[k] for k in unpaid.keys()]
    fig, ax = plt.subplots(1, 1, figsize=(25, 25))
    ax.scatter(paid_AGG, unpaid_AGG)
    ax.set_title("Scatterplot of Paid vs. Unpaid Claims aggregated by ProviderID", fontsize=42)
    ax.set_xlabel("Number of Paid Claims", fontsize=30)
    ax.set_ylabel("Number of Unpaid Claims", fontsize=30)
    ax.set_xlim(-5, 2000)
    ax.set_ylim(-5, 15000)

    try:
        plt.savefig(output_dir + '/' + filename, bbox_inches='tight')
    except IOError:
        os.mkdir(output_dir)
        plt.savefig(output_dir + '/' + filename, bbox_inches='tight')

    print(
        "Scatterplot displaying number of unpaid versus paid claims is saved in working directory as " + filename)
    print(
        "The graph suggests that there are many more unpaid J-code claims (45,057) than paid J-code claims (5,972). Note the different scales on the axes: y-axis (unpaid) is set to a max of 15,000 while x-axis (paid) is set to a max of 2,500 for better resolution. Scale of y-axis is 7.5x the scale of x-axis, which is approximately how many times greater the number of unpaid J-code claims is than the number of paid J-code claims")
    print(
        "The concerning behavior from the graph is that the number of unpaid claims are much greater than the number of paid claims. This suggests an overwhelming medical debt in this data population that is possibly looming over the entire healthcare industry. Healthcare providers will need to take significant steps to improve their claim collection process as this can be a major operational issue.")

    unpaid_pct = 0
    paid_pct = 0

    for row in Jcodes:
        if int(float(row[columnMap["ProviderPaymentAmount"]])) == 0:
            unpaid_pct += 1
        else:
            paid_pct += 1
    total = unpaid_pct + paid_pct
    percent_unpaid = float(unpaid_pct / total * 100)
    print("The percentage of unpaid J-code claim lines is: " + str(round(percent_unpaid, 2)) + '%')

output_dir = 'output' + '/'
output_dir_best = output_dir + 'BestResult'
output_dir_GridSearchResults = output_dir + '/' + 'CompleteResults'

with open('claim.sample.csv', 'r') as f:
    print(f.readline())
    print(f.readline())

names = ["V1","Claim.Number","Claim.Line.Number",
         "Member.ID","Provider.ID","Line.Of.Business.ID",
         "Revenue.Code","Service.Code","Place.Of.Service.Code",
         "Procedure.Code","Diagnosis.Code","Claim.Charge.Amount",
         "Denial.Reason.Code","Price.Index","In.Out.Of.Network",
         "Reference.Index","Pricing.Index","Capitation.Index",
         "Subscriber.Payment.Amount","Provider.Payment.Amount",
         "Group.Index","Subscriber.Index","Subgroup.Index",
         "Claim.Type","Claim.Subscriber.Type","Claim.Pre.Prince.Index",
         "Claim.Current.Status","Network.ID","Agreement.ID"]

types = ['S8', 'f8', 'i4', 'i4', 'S14', 'S6', 'S6', 'S6', 'S4', 'S9', 'S7', 'f8',
         'S5', 'S3', 'S3', 'S3', 'S3', 'S3', 'f8', 'f8', 'i4', 'i4', 'i4', 'S3',
         'S3', 'S3', 'S4', 'S14', 'S14']

CLAIMS = np.genfromtxt('claim.sample.csv', dtype=types, delimiter=',', names=True,
                       usecols=[0,1,2,3,4,5,
                                6,7,8,9,10,11,
                                12,13,14,15,16,
                                17,18,19,20,21,
                                22,23,24,25,26,
                                27,28])

Jcodes = get_JCodes_with_claims(CLAIMS)

get_provider_amounts(Jcodes)

provider_payments(Jcodes,5)

columns = CLAIMS.dtype.names
columnMap = {k:v for v, k in enumerate(columns)}
unpaid_claims = Counter()
paid_claims = Counter()

for row in Jcodes:
    if int(float(row[columnMap["ProviderPaymentAmount"]])) == 0:
        unpaid_claims[row[columnMap["ProviderID"]]] += 1
    else:
        paid_claims[row[columnMap["ProviderID"]]] += 1

paid_vs_unpaid(paid_claims,unpaid_claims)

model_cols = []
numeric_cols = ['SubscriberPaymentAmount', 'ClaimChargeAmount']
dummy_cols = {'ProviderID':{}, 'ServiceCode':{},'InOutOfNetwork':{},'NetworkID':{},'AgreementID':{}, 'DenialReasonCode':{}, 'PricingIndex':{}, 'ReferenceIndex':{}, 'ClaimPrePrinceIndex':{}, 'ClaimType':{},'ClaimSubscriberType':{}, 'ClaimCurrentStatus':{}, 'ProcedureCode':{}, 'RevenueCode':{}, 'DiagnosisCode':{}}
model_columns = []

for rownum, row in enumerate(Jcodes):
    for col in dummy_cols.keys():
        try:
            dummy_cols[col][rownum].add(row[columnMap[col]])
        except KeyError:
            dummy_cols[col][row[columnMap[col]]] = {rownum}
for colname, dumdict in dummy_cols.items():
    for dummy in dumdict.keys():
        model_columns.append("{}${}".format(colname, dummy.decode('UTF-8')))

classifierDict = {'RandomForestClassifier':{
    'min_samples_split': [2,3,5,6],
    'n_jobs': [3,5]},
    'LogisticRegression':{'tol': [0.001,0.0011,.005,.0055]},
          }

clfDict = {'RandomForestClassifier':{
    'min_samples_split': [5,6],
    'n_jobs': [3,5]},
    'LogisticRegression':{'tol': [0.001,.0055]},
    'KNeighborsClassifier': {
        'n_neighbors': [20,30],
        'algorithm': ['auto','ball_tree','brute'],
        'p': [5]},
    'QuadraticDiscriminantAnalysis':{'tol':[0.001,.0055],
    'store_covariance':[False,True]},
    'DecisionTreeClassifier':{'max_depth':[1,5]}
          }

classifierList = {RandomForestClassifier,LogisticRegression,DecisionTreeClassifier}
clfList = {RandomForestClassifier,LogisticRegression,DecisionTreeClassifier,KNeighborsClassifier,QuadraticDiscriminantAnalysis}

model_df = []
for idx, row in enumerate(Jcodes):
    ith_row = []
    for colname in model_columns:
        col, val = colname.split("$")
        if idx in dummy_cols.get(col,val):
            ith_row.append(1)
        else:
            ith_row.append(0)
    for numeric in numeric_cols:
        ith_row.append(float(row[columnMap[numeric]]))

    if float(row[columnMap['ProviderPaymentAmount']]) > 0.0:
        ith_row.append(0.0)
    else:
        ith_row.append(1.0)
    model_df.append(ith_row)

for numeric_col in numeric_cols:
    model_columns.append(numeric_col)
np_df = np.array(model_df)

X = np_df[:, :-1]
y = np_df[:, -1]
n_folds = 5
data = (X,y,n_folds)

grid_search = DeathToGridSearch(clfDict,clfList,data)
grid_search_classifiers = DeathToGridSearch(classifierDict,classifierList,data)

classifier_accuracy, importance_dict = grid_search_classifiers.run_gridsearch_classifiers()
grid_search_classifiers.get_best_score(classifier_accuracy)

accuracy_dict = grid_search.run_gridsearch()
grid_search.get_best_score(accuracy_dict)
