import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arff
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def print_help():
    print("Usage: python3 project.py [DT, RF, SVM, NB, VS, ALL] [file]")
    print("")
    print("DT: Decision Tree")
    print("RF: Random Forest")
    print("SVM: Support vector machine")
    print("NB: Naive Bayes")
    print("VS: only show data visualization")
    print("file: please input file path")
    print("")
    exit(-1) 

def variance_threshold_selector(traindata, threshold):
    selector = VarianceThreshold(threshold)
    selector.fit(traindata)
    return traindata[traindata.columns[selector.get_support(indices=True)]]

if len(sys.argv) < 2:
    print_help()
if (sys.argv[1] != 'DT' and sys.argv[1] != 'RF' and sys.argv[1] != 'SVM' and sys.argv[1] != 'NB' and sys.argv[1] != 'VS' and sys.argv[1] != 'ALL'):
    print_help()

file_name = sys.argv[2]

dataset = arff.load(open(file_name), 'rb')
# print(dataset['attributes'])
att_list = []
for att in dataset['attributes']:
    att_list.append(att[0])
Traindata = np.array(dataset['data'])
Traindata = pd.DataFrame(Traindata, columns=att_list)
# print(Traindata)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


target = Traindata.values[:, len(Traindata.columns) - 1]
# target = target.astype('int')
Traindata.drop(att_list[-1], axis=1, inplace=True)
# print(Traindata)

# ---------------- visualize ----------------
if sys.argv[1] == 'VS':
    Traindata_visual = Traindata.copy()
    plt.subplot(2, 1, 1)
    pd.value_counts(Traindata_visual['having_IP_Address']).plot.bar()
    plt.title('having_IP_Address')
    plt.subplot(2, 1, 2)
    pd.value_counts(Traindata_visual['URL_Length']).plot.bar()
    plt.title('URL_Length')
    plt.show()
    exit(-1)
# ---------------- visualize ----------------

# Traindata = variance_threshold_selector(Traindata, 0.1)
# print(Traindata)

scaler = StandardScaler()
scaler.fit(Traindata)
Traindata = pd.DataFrame(scaler.transform(Traindata), index=Traindata.index, columns=Traindata.columns)
# print(Traindata)

feature = Traindata.values

feature, feature_test, target, target_test = train_test_split(feature, target, test_size = 0.2, random_state = 42)

kf = KFold(n_splits=10, shuffle=True)


if sys.argv[1] == 'DT' or sys.argv[1] == 'ALL':
    pipe = Pipeline([
        ('feature_selection', PCA()),
        ('model', DecisionTreeClassifier())])
    params = {'model__criterion': ['gini', 'entropy'],
              'model__random_state': [42, 100]}
    best = GridSearchCV(pipe, params, cv=kf, n_jobs=32)
    best.fit(feature, target)

    target_pred = best.predict(feature_test)

    print("--------- By Decision Tree ---------")
    print("By micro average")
    print("Recall: ", recall_score(target_test, target_pred, average='micro'))
    print("Precision: ", precision_score(target_test, target_pred, average='micro'))
    print("F1-Score: ", f1_score(target_test, target_pred, average='micro'))
    print("")
    print("By macro average")
    print("Recall: ", recall_score(target_test, target_pred, average='macro'))
    print("Precision: ", precision_score(target_test, target_pred, average='macro'))
    print("F1-Score: ", f1_score(target_test, target_pred, average='macro'))
    print("")
    print("Confusion matrix:\n", confusion_matrix(target_test, target_pred))
    print("------------------------------------\n")
    # disp = plot_confusion_matrix(clf, feature_test, target_test, 
    #                              cmap=plt.cm.Blues, display_labels=label_names, normalize='true', values_format=".4f")
    # disp.ax_.set_title('Confusion matrix (Decision Tree)')
    # plt.show()
    
if sys.argv[1] == 'RF' or sys.argv[1] == 'ALL':
    pipe = Pipeline([
        ('feature_selection', PCA()),
        ('model', RandomForestClassifier())])
    params = {'model__criterion': ['gini', 'entropy'],
              'model__n_estimators': [100, 300]}
    best = GridSearchCV(pipe, params, cv=kf, n_jobs=32)
    best.fit(feature, target)

    target_pred = best.predict(feature_test)
    print("--------- By Random Forest ---------")
    print("By micro average")
    print("Recall: ", recall_score(target_test, target_pred, average='micro'))
    print("Precision: ", precision_score(target_test, target_pred, average='micro'))
    print("F1-Score: ", f1_score(target_test, target_pred, average='micro'))
    print("")
    print("By macro average")
    print("Recall: ", recall_score(target_test, target_pred, average='macro'))
    print("Precision: ", precision_score(target_test, target_pred, average='macro'))
    print("F1-Score: ", f1_score(target_test, target_pred, average='macro'))
    print("")
    print("Confusion matrix:\n", confusion_matrix(target_test, target_pred))
    print("------------------------------------\n")
    # disp = plot_confusion_matrix(clf, feature_test, target_test, 
    #                              cmap=plt.cm.Blues, display_labels=label_names, normalize='true', values_format=".4f")
    # disp.ax_.set_title('Confusion matrix (Random Forest)')
    # plt.show()

if sys.argv[1] == 'SVM' or sys.argv[1] == 'ALL':
    pipe = Pipeline([
        ('feature_selection', PCA()),
        ('model', svm.SVC())])
    params = {'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    best = GridSearchCV(pipe, params, cv=kf, n_jobs=32)
    best.fit(feature, target)

    target_pred = best.predict(feature_test)
    print("---------- By SVM ----------")
    print("By micro average")
    print("Recall: ", recall_score(target_test, target_pred, average='micro'))
    print("Precision: ", precision_score(target_test, target_pred, average='micro'))
    print("F1-Score: ", f1_score(target_test, target_pred, average='micro'))
    print("")
    print("By macro average")
    print("Recall: ", recall_score(target_test, target_pred, average='macro'))
    print("Precision: ", precision_score(target_test, target_pred, average='macro'))
    print("F1-Score: ", f1_score(target_test, target_pred, average='macro'))
    print("")
    print("Confusion matrix:\n", confusion_matrix(target_test, target_pred))
    print("------------------------------------\n")
    # disp = plot_confusion_matrix(clf, feature_test, target_test, 
    #                              cmap=plt.cm.Blues, display_labels=label_names, normalize='true', values_format=".4f")
    # disp.ax_.set_title('Confusion matrix (NB)')
    # plt.show()

if sys.argv[1] == 'NB' or sys.argv[1] == 'ALL':
    pipe = Pipeline([
        ('feature_selection', PCA()),
        ('model', GaussianNB())])
    params = {'model__priors': [0.15, 0.25, 0.35, 0.5, None]}
    best = GridSearchCV(pipe, params, cv=kf, n_jobs=32)
    best.fit(feature, target)

    target_pred = best.predict(feature_test)
    print("---------- By Naive Bayes ----------")
    print("By micro average")
    print("Recall: ", recall_score(target_test, target_pred, average='micro'))
    print("Precision: ", precision_score(target_test, target_pred, average='micro'))
    print("F1-Score: ", f1_score(target_test, target_pred, average='micro'))
    print("")
    print("By macro average")
    print("Recall: ", recall_score(target_test, target_pred, average='macro'))
    print("Precision: ", precision_score(target_test, target_pred, average='macro'))
    print("F1-Score: ", f1_score(target_test, target_pred, average='macro'))
    print("")
    print("Confusion matrix:\n", confusion_matrix(target_test, target_pred))
    print("------------------------------------\n")
    # disp = plot_confusion_matrix(clf, feature_test, target_test, 
    #                              cmap=plt.cm.Blues, display_labels=label_names, normalize='true', values_format=".4f")
    # disp.ax_.set_title('Confusion matrix (NB)')
    # plt.show()