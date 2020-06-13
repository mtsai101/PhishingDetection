import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

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

pd.set_option('display.float_format', lambda x: '%.2f' % x)
dataset = pd.read_csv('../dataset/UNB_Phishing.csv')
dataset = dataset.fillna(-1)
imp = SimpleImputer(missing_values=np.inf, strategy='constant',fill_value=1)
imp.fit_transform(dataset)

le = LabelEncoder()
dataset['URL_Type_obf_Type'] = le.fit_transform(dataset['URL_Type_obf_Type'])
Traindata = dataset.astype('float64')
target = Traindata.values[:, len(Traindata.columns) - 1]
Traindata.drop('URL_Type_obf_Type', axis=1, inplace=True)
class_list = list(le.classes_)
row_format ="{:>15}" * (len(class_list) + 1)

for k,i in enumerate(Traindata['argPathRatio']):
    if np.isinf(Traindata['argPathRatio'][k]): 
        Traindata['argPathRatio'][k] = 2.0

# ---------------- visualize ----------------
# class_num = Traindata['URL_Type_obf_Type'].value_counts()
# class_num.plot(kind='bar', title='Count (class)')
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
    print("Confusion matrix:")
    print(row_format.format("", *class_list))
    for team, row in zip(class_list, confusion_matrix(target_test,target_pred)):
        print(row_format.format(team, *row))
    print(classification_report(target_test, target_pred,target_names=class_list))
    print("Misclassified sample: %d" % (target_test != target_pred).sum())
    # print("By micro average")
    # print("Recall: ", recall_score(target_test, target_pred, average='micro'))
    # print("Precision: ", precision_score(target_test, target_pred, average='micro'))
    # print("F1-Score: ", f1_score(target_test, target_pred, average='micro'))
    # print("")
    # print("By macro average")
    # print("Recall: ", recall_score(target_test, target_pred, average='macro'))
    # print("Precision: ", precision_score(target_test, target_pred, average='macro'))
    # print("F1-Score: ", f1_score(target_test, target_pred, average='macro'))
    # print("")
    # print("Confusion matrix:\n", confusion_matrix(target_test, target_pred))
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
    print("Confusion matrix:")
    print(row_format.format("", *class_list))
    for team, row in zip(class_list, confusion_matrix(target_test,target_pred)):
        print(row_format.format(team, *row))
    print(classification_report(target_test, target_pred,target_names=class_list))
    print("Misclassified sample: %d" % (target_test != target_pred).sum())
    # print("By micro average")
    # print("Recall: ", recall_score(target_test, target_pred, average='micro'))
    # print("Precision: ", precision_score(target_test, target_pred, average='micro'))
    # print("F1-Score: ", f1_score(target_test, target_pred, average='micro'))
    # print("")
    # print("By macro average")
    # print("Recall: ", recall_score(target_test, target_pred, average='macro'))
    # print("Precision: ", precision_score(target_test, target_pred, average='macro'))
    # print("F1-Score: ", f1_score(target_test, target_pred, average='macro'))
    # print("")
    # print("Confusion matrix:\n", confusion_matrix(target_test, target_pred))
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
    print("Confusion matrix:")
    print(row_format.format("", *class_list))
    for team, row in zip(class_list, confusion_matrix(target_test,target_pred)):
        print(row_format.format(team, *row))
    print(classification_report(target_test, target_pred,target_names=class_list))
    print("Misclassified sample: %d" % (target_test != target_pred).sum())
    # print("By micro average")
    # print("Recall: ", recall_score(target_test, target_pred, average='micro'))
    # print("Precision: ", precision_score(target_test, target_pred, average='micro'))
    # print("F1-Score: ", f1_score(target_test, target_pred, average='micro'))
    # print("")
    # print("By macro average")
    # print("Recall: ", recall_score(target_test, target_pred, average='macro'))
    # print("Precision: ", precision_score(target_test, target_pred, average='macro'))
    # print("F1-Score: ", f1_score(target_test, target_pred, average='macro'))
    # print("")
    # print("Confusion matrix:\n", confusion_matrix(target_test, target_pred))
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
    print("Confusion matrix:")
    print(row_format.format("", *class_list))
    for team, row in zip(class_list, confusion_matrix(target_test,target_pred)):
        print(row_format.format(team, *row))
    print(classification_report(target_test, target_pred,target_names=class_list))
    print("Misclassified sample: %d" % (target_test != target_pred).sum())
    # print("By micro average")
    # print("Recall: ", recall_score(target_test, target_pred, average='micro'))
    # print("Precision: ", precision_score(target_test, target_pred, average='micro'))
    # print("F1-Score: ", f1_score(target_test, target_pred, average='micro'))
    # print("")
    # print("By macro average")
    # print("Recall: ", recall_score(target_test, target_pred, average='macro'))
    # print("Precision: ", precision_score(target_test, target_pred, average='macro'))
    # print("F1-Score: ", f1_score(target_test, target_pred, average='macro'))
    # print("")
    # print("Confusion matrix:\n", confusion_matrix(target_test, target_pred))
    print("------------------------------------\n")
    # disp = plot_confusion_matrix(clf, feature_test, target_test, 
    #                              cmap=plt.cm.Blues, display_labels=label_names, normalize='true', values_format=".4f")
    # disp.ax_.set_title('Confusion matrix (NB)')
    # plt.show()