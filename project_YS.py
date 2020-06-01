import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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

def print_help():
    print("Usage: python3 project.py [DT, RF, NB, VS]")
    print("")
    print("DT: Decision Tree")
    print("RF: Random Forest")
    print("NB: Naive Bayes")
    print("VS: only show data visualization")
    print("")
    exit(-1) 

def variance_threshold_selector(traindata, testdata, threshold=0.01):
    selector = VarianceThreshold(threshold)
    selector.fit(traindata)
    return traindata[traindata.columns[selector.get_support(indices=True)]],\
           testdata[testdata.columns[selector.get_support(indices=True)]]

if len(sys.argv) < 2:
    print_help()
if (sys.argv[1] != 'DT' and sys.argv[1] != 'RF' and sys.argv[1] != 'KNN' and sys.argv[1] != 'NB' and sys.argv[1] != 'VS'):
    print_help()

dataset = arff.load(open('Training Dataset.arff'), 'rb')
print(dataset['attributes'])
att_list = []
for att in dataset['attributes']:
    att_list.append(att[0])
Traindata = np.array(dataset['data'])
Traindata = pd.DataFrame(Traindata, columns=att_list)
print(Traindata)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# target = Traindata.values[:, len(Traindata.columns) - 1]
# target = target.astype('int')
# Traindata.drop(len(Traindata.columns) - 1, axis=1, inplace=True)

# # ---------------- visualize ----------------
# if sys.argv[1] == 'VS':
#     Traindata_visual = Traindata.copy()
#     Traindata_visual.columns = featureList
#     # features = ['count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']
#     # features = ['count', 'srv_count']
#     # print(Traindata_visual[features].describe())
#     plt.subplot(2, 1, 1)
#     pd.value_counts(Traindata_visual['protocol_type']).plot.bar()
#     plt.title('protocol_type')
#     plt.subplot(2, 1, 2)
#     pd.value_counts(Traindata_visual['service']).plot.bar()
#     plt.title('service')
#     # Traindata_visual[features].hist(figsize=(20, 8), grid=False)
#     plt.show()
#     exit(-1)
# # ---------------- visualize ----------------

# # ------------ label encoding example ------------
# # labelencoder = LabelEncoder()
# # Traindata[1] = labelencoder.fit_transform(Traindata[1])
# # Traindata[2] = labelencoder.fit_transform(Traindata[2])
# # Traindata[3] = labelencoder.fit_transform(Traindata[3])
# # ------------ label encoding example ------------

# temp = pd.get_dummies(pd.concat([Traindata,Testdata],keys=[0,1]))
# Traindata, Testdata = temp.xs(0), temp.xs(1)
# # print(Traindata)
# # print(target)
# # print(Testdata)
# # print(target_test)

# # Traindata = pd.get_dummies(Traindata)

# Traindata, Testdata = variance_threshold_selector(Traindata, Testdata, 0.01)
# # print(Traindata)
# # print(Testdata)

# scaler = StandardScaler()
# scaler.fit(Traindata)
# Traindata = pd.DataFrame(scaler.transform(Traindata), index=Traindata.index, columns=Traindata.columns)
# Testdata = pd.DataFrame(scaler.transform(Testdata), index=Testdata.index, columns=Testdata.columns)
# # print(Traindata)
# # print(Testdata)

# feature = Traindata.values
# feature_test = Testdata.values

# kf = KFold(n_splits=5, shuffle=True)

# if sys.argv[1] == 'DT':
#     # clf = DecisionTreeClassifier(criterion = "entropy", random_state = 42)
#     clf = DecisionTreeClassifier()
#     params = {'criterion': ['gini', 'entropy'],
#               'max_features': ['auto', 'sqrt'],
#               'max_depth': [10, 20, 30, 40, 50, None],
#               'min_samples_split': [2, 5, 7, 9],
#               'min_samples_leaf': [1, 3, 5],
#               'random_state': [42, 100]}
#     clf_random = RandomizedSearchCV(estimator = clf, param_distributions = params, 
#                                     n_iter = 100, cv = kf, verbose=2, random_state=42, n_jobs = -1)
#     clf_random.fit(feature, target)
#     best_random = clf_random.best_estimator_
#     target_pred = best_random.predict(feature_test)
#     # target_pred = clf.predict(feature_test)
#     print("--------- By Decision Tree ---------")
#     print("By micro average")
#     print("Recall: ", recall_score(target_test, target_pred, average='micro'))
#     print("Precision: ", precision_score(target_test, target_pred, average='micro'))
#     print("F1-Score: ", f1_score(target_test, target_pred, average='micro'))
#     print("")
#     print("By macro average")
#     print("Recall: ", recall_score(target_test, target_pred, average='macro'))
#     print("Precision: ", precision_score(target_test, target_pred, average='macro'))
#     print("F1-Score: ", f1_score(target_test, target_pred, average='macro'))
#     print("")
#     print("Confusion matrix:\n", confusion_matrix(target_test, target_pred))
#     print("------------------------------------\n")
#     # disp = plot_confusion_matrix(clf, feature_test, target_test, 
#     #                              cmap=plt.cm.Blues, display_labels=label_names, normalize='true', values_format=".4f")
#     # disp.ax_.set_title('Confusion matrix (Decision Tree)')
#     # plt.show()
    
# if sys.argv[1] == 'RF':
#     # clf = RandomForestClassifier(criterion = "entropy", random_state = 100, n_estimators = 300)
#     clf = RandomForestClassifier()
#     params = {'criterion': ['gini', 'entropy'],
#               'n_estimators': [100, 200, 300, 400, 500],
#               'max_features': ['auto', 'sqrt'],
#               'max_depth': [10, 20, 30, 40, 50, None],
#               'min_samples_split': [2, 5, 7, 9],
#               'min_samples_leaf': [1, 3, 5]}
#     clf_random = RandomizedSearchCV(estimator = clf, param_distributions = params, 
#                                     n_iter = 100, cv = kf, verbose=2, random_state=42, n_jobs = -1)
#     clf_random.fit(feature, target)
#     best_random = clf_random.best_estimator_
#     target_pred = best_random.predict(feature_test)
#     # target_pred = clf.predict(feature_test)
#     print("--------- By Random Forest ---------")
#     print("By micro average")
#     print("Recall: ", recall_score(target_test, target_pred, average='micro'))
#     print("Precision: ", precision_score(target_test, target_pred, average='micro'))
#     print("F1-Score: ", f1_score(target_test, target_pred, average='micro'))
#     print("")
#     print("By macro average")
#     print("Recall: ", recall_score(target_test, target_pred, average='macro'))
#     print("Precision: ", precision_score(target_test, target_pred, average='macro'))
#     print("F1-Score: ", f1_score(target_test, target_pred, average='macro'))
#     print("")
#     print("Confusion matrix:\n", confusion_matrix(target_test, target_pred))
#     print("------------------------------------\n")
#     # disp = plot_confusion_matrix(clf, feature_test, target_test, 
#     #                              cmap=plt.cm.Blues, display_labels=label_names, normalize='true', values_format=".4f")
#     # disp.ax_.set_title('Confusion matrix (Random Forest)')
#     # plt.show()

# if sys.argv[1] == 'NB':
#     clf = GaussianNB()
#     params = {'priors': [0.15, 0.25, 0.35, 0.5, None],
#               'var_smoothing': [ i*(1e-09) for i in range(10, 100)]}
#     clf_random = RandomizedSearchCV(estimator = clf, param_distributions = params, 
#                                     n_iter = 100, cv = kf, verbose=2, random_state=42, n_jobs = -1)
#     clf_random.fit(feature, target)
#     best_random = clf_random.best_estimator_
#     target_pred = best_random.predict(feature_test)
#     # target_pred = clf.predict(feature_test)
#     print("---------- By Naive Bayes ----------")
#     print("By micro average")
#     print("Recall: ", recall_score(target_test, target_pred, average='micro'))
#     print("Precision: ", precision_score(target_test, target_pred, average='micro'))
#     print("F1-Score: ", f1_score(target_test, target_pred, average='micro'))
#     print("")
#     print("By macro average")
#     print("Recall: ", recall_score(target_test, target_pred, average='macro'))
#     print("Precision: ", precision_score(target_test, target_pred, average='macro'))
#     print("F1-Score: ", f1_score(target_test, target_pred, average='macro'))
#     print("")
#     print("Confusion matrix:\n", confusion_matrix(target_test, target_pred))
#     print("------------------------------------\n")
#     # disp = plot_confusion_matrix(clf, feature_test, target_test, 
#     #                              cmap=plt.cm.Blues, display_labels=label_names, normalize='true', values_format=".4f")
#     # disp.ax_.set_title('Confusion matrix (NB)')
#     # plt.show()

# if sys.argv[1] == 'KNN':
#     knn = KNeighborsClassifier(n_neighbors=5)
#     knn.fit(feature, target)
#     target_pred = knn.predict(feature_test)
#     print("-------------- By KNN --------------")
#     print("By micro average")
#     print("Recall: ", recall_score(target_test, target_pred, average='micro'))
#     print("Precision: ", precision_score(target_test, target_pred, average='micro'))
#     print("F1-Score: ", f1_score(target_test, target_pred, average='micro'))
#     print("")
#     print("By macro average")
#     print("Recall: ", recall_score(target_test, target_pred, average='macro'))
#     print("Precision: ", precision_score(target_test, target_pred, average='macro'))
#     print("F1-Score: ", f1_score(target_test, target_pred, average='macro'))
#     print("")
#     print("Confusion matrix:\n", confusion_matrix(target_test, target_pred))
#     print("------------------------------------\n")
#     # disp = plot_confusion_matrix(knn, feature_test, target_test, 
#     #                              cmap=plt.cm.Blues, display_labels=label_names, normalize='true', values_format=".4f")
#     # disp.ax_.set_title('Confusion matrix (KNN)')
#     # plt.show()

# # # x meas feature, y means target
# # kf = KFold(n_splits = 3)
# # # x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size = 0.1, random_state = 42)
# # # clf_entropy.fit(x_train, y_train)

# # # y_pred = clf_entropy.predict(x_test)
