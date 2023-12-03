import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,TimeSeriesSplit,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,ConfusionMatrixDisplay,confusion_matrix,recall_score,precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from imblearn.metrics import specificity_score
from imblearn.over_sampling import ADASYN
import warnings

'Starting Time'
import time
start = time.time()

warnings.filterwarnings("ignore")
n_jobs = -1

shuffle = False
gc.enable()

tscv = TimeSeriesSplit(n_splits=12)
df = (pd.read_csv('classification.csv'))
df['classification'] = np.select([df['Y'].between(0,50),
                                 df['Y'].between(51,300)]
                                ,['Healthy','Unhealthy'])

y = df['classification']
features_list = ['Year','CO Mean', 'SO2 Mean', 'NO2 Mean', 'NO2 1st Max Hour', 'AQI',
       'O3_AQI_label','days_since_start']
df.drop(columns=[col for col in df.keys() if col not in features_list],inplace=True)
X = df.copy()

"""
=====================
Transformations
=====================
"""
co_mean_std = StandardScaler()
co_mean_std.fit(X['CO Mean'].to_numpy().reshape(len(X['CO Mean']), -1))
X['CO Mean'] = co_mean_std.fit_transform(X['CO Mean'].to_numpy().reshape(len(X['CO Mean']),-1))

so2_mean_std = StandardScaler()
so2_mean_std.fit(X['SO2 Mean'].to_numpy().reshape(len(X['CO Mean']), -1))
X['SO2 Mean'] = so2_mean_std.fit_transform(X['SO2 Mean'].to_numpy().reshape(len(X['CO Mean']),-1))

no2_mean_std = StandardScaler()
no2_mean_std.fit(X['NO2 Mean'].to_numpy().reshape(len(X['CO Mean']), -1))
X['NO2 Mean'] = no2_mean_std.fit_transform(X['NO2 Mean'].to_numpy().reshape(len(X['CO Mean']),-1))

no2_1st_max_hr_std = StandardScaler()
no2_1st_max_hr_std.fit(X['NO2 1st Max Hour'].to_numpy().reshape(len(X['CO Mean']), -1))
X['NO2 1st Max Hour'] = no2_1st_max_hr_std.fit_transform(X['NO2 1st Max Hour'].to_numpy().reshape(len(X['CO Mean']),-1))

aqi_std = StandardScaler()
aqi_std.fit(X['AQI'].to_numpy().reshape(len(X['CO Mean']), -1))
X['AQI'] = aqi_std.fit_transform(X['AQI'].to_numpy().reshape(len(X['CO Mean']),-1))

days_start_std = StandardScaler()
days_start_std.fit(X['days_since_start'].to_numpy().reshape(len(X['CO Mean']), -1))
X['days_since_start'] = days_start_std.fit_transform(X['days_since_start'].to_numpy().reshape(len(X['CO Mean']),-1))

year_std = StandardScaler()
year_std.fit(X['Year'].to_numpy().reshape(len(X['CO Mean']), -1))
X['Year'] = year_std.fit_transform(X['Year'].to_numpy().reshape(len(X['CO Mean']),-1))

le = LabelEncoder()
le.fit(X['O3_AQI_label'])
X['O3_AQI_label'] = le.fit_transform(X['O3_AQI_label'].to_numpy().reshape(len(X['CO Mean']),-1))

le_y = LabelEncoder()
le_y.fit(y)
y = le_y.fit_transform(y.to_numpy().reshape(len(X['CO Mean']),-1))

"""
=========================|
Data Imbalance Used ADASYN 
=========================|
"""
sns.countplot(x=y)
plt.title('Countplot of Target')
plt.tight_layout()
plt.show()

oversample = ADASYN(sampling_strategy='auto',n_jobs=n_jobs)#n_neighbors=5
X, y = oversample.fit_resample(X, y)
X['y_class'] = y

X['Year'] = year_std.inverse_transform(X['Year'].to_numpy().reshape(len(X['CO Mean']),-1))
X['days_since_start'] = days_start_std.inverse_transform(X['days_since_start'].to_numpy().reshape(len(X['CO Mean']),-1))

X.sort_values(['days_since_start','Year'],axis=0,inplace=True)
y = X['y_class']

X.drop(columns=['Year','y_class'],inplace=True,axis=1)
X['days_since_start'] = days_start_std.fit_transform(X['days_since_start'].to_numpy().reshape(len(X['CO Mean']),-1))

sns.countplot(x=y)
plt.title('Countplot of Target')
plt.tight_layout()
plt.show()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2 ,shuffle=shuffle)

"""
====================
Logistic Regression
====================
"""
print('================================================================================================')

print('Logistic Regression')
lg = LogisticRegression(fit_intercept=True)
lg.fit(X_train,y_train)
pred = lg.predict(X_test)
print(f'ACCURACY: {round(accuracy_score(y_test,pred),3)}')
print(f'F1: {round(f1_score(y_test,pred),3)}')
print(f'Precision: {round(precision_score(y_test,pred),3)}')
print(f'Recall: {round(recall_score(y_test,pred),3)}')
print(f'Specificity:{round(specificity_score(y_test,pred),3)}')
print('================================================================================================')
y_proba_lg = lg.predict_proba(X_test)[::,-1]


"""
=================|
Confusion Matrix |
=================|
"""
matrix = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=lg.classes_)
disp.plot()
plt.title('Logisitic Regression Confusion Matrix')
plt.show()
del lg

"""
====================
Decision Tree
====================
"""
print('================================================================================================')
'Base'
dt = DecisionTreeClassifier(max_depth=10)
dt.fit(X_train,y_train)
pred = dt.predict(X_test)
range = [0.0,0.01,0.02,0.03]
print('DECISION TREE')
print(f'ACCURACY: {round(accuracy_score(y_test,pred),3)}')
print(f'F1: {round(f1_score(y_test,pred),3)}')
print(f'Precision: {round(precision_score(y_test,pred),3)}')
print(f'Recall: {round(recall_score(y_test,pred),3)}')
print(f'Specificity:{round(specificity_score(y_test,pred),3)}')
print('================================================================================================')
y_proba_dt = dt.predict_proba(X_test)[::,-1]

"""
================
Confusion Matrix base
================
"""
matrix = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=dt.classes_)
disp.plot()
plt.title('Decision Tree Confusion Matrix')
plt.show()
del dt

param_grid = {'criterion':['gini', 'entropy','log_loss'],
              'splitter':['best', 'random'],
              'max_features':['auto', 'sqrt', 'log2'],
              'min_weight_fraction_leaf':range,
              'min_samples_leaf':np.arange(0,.05,.01),
              'min_impurity_decrease':range,
              'min_samples_split':np.arange(2,5,1)
              }

dt_grid = GridSearchCV(DecisionTreeClassifier(),
                       cv=tscv,param_grid=param_grid,n_jobs=n_jobs,verbose=1)

dt_grid.fit(X,y.ravel())
dt_grid_result = dt_grid.best_params_
print(dt_grid.best_params_)
dt_pre = DecisionTreeClassifier(
                        criterion=dt_grid_result['criterion'],
                        splitter=dt_grid_result['splitter'],max_features=dt_grid_result['max_features'],
    min_weight_fraction_leaf=dt_grid_result['min_weight_fraction_leaf'],
    min_samples_leaf=dt_grid_result['min_samples_leaf'],min_impurity_decrease=dt_grid_result['min_impurity_decrease'],
min_samples_split=dt_grid_result['min_samples_split'])

dt_pre.fit(X_train,y_train)
pred = dt_pre.predict(X_test)
print('========================')
print('Pre Pruned Grid Search Decision Tree')
print(f'ACCURACY: {round(accuracy_score(y_test,pred),3)}')
print(f'F1: {round(f1_score(y_test,pred),3)}')
print(f'Precision: {round(precision_score(y_test,pred),3)}')
print(f'Recall: {round(recall_score(y_test,pred),3)}')
print(f'Specificity:{round(specificity_score(y_test,pred),3)}')
print('================================================================================================')
y_proba_dt_pre = dt_pre.predict_proba(X_test)[::,-1]

"""
====================
Confusion Matrix pre
====================
"""
matrix = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=dt_pre.classes_)
disp.plot()
plt.title('Pre Pruned Decision Tree Confusion Matrix')
plt.show()

param_grid = {'ccp_alpha':np.arange(0,.05,.001)}
dt_grid = GridSearchCV(DecisionTreeClassifier(),
                       cv=tscv, param_grid=param_grid, n_jobs=n_jobs, verbose=1)

dt_grid.fit(X, y.ravel())
dt_grid_result = dt_grid.best_params_
print(dt_grid.best_params_)
dt_post = DecisionTreeClassifier(ccp_alpha=dt_grid_result['ccp_alpha'])
dt_post.fit(X_train, y_train)
pred = dt_post.predict(X_test)
print('========================')
print('Post Pruned Grid Search Decision Tree')
print(f'ACCURACY: {round(accuracy_score(y_test, pred), 3)}')
print(f'F1: {round(f1_score(y_test, pred), 3)}')
print(f'Precision: {round(precision_score(y_test,pred),3)}')
print(f'Recall: {round(recall_score(y_test,pred),3)}')
print(f'Specificity:{round(specificity_score(y_test,pred),3)}')
print('================================================================================================')
y_proba_dt_post = dt_post.predict_proba(X_test)[::,-1]

"""
====================
Confusion Matrix post
====================
"""
matrix = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=dt_post.classes_)
disp.plot()
plt.title('Post Pruned Decision Tree Confusion Matrix')
plt.show()
del dt_pre
del dt_post

"""
====================
KNN
====================
"""
print('================================================================================================')
knn = KNeighborsClassifier(n_jobs=n_jobs)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print('KNN')
print(f'ACCURACY: {round(accuracy_score(y_test,pred),3)}')
print(f'F1: {round(f1_score(y_test,pred),3)}')
print(f'Precision: {round(precision_score(y_test,pred),3)}')
print(f'Recall: {round(recall_score(y_test,pred),3)}')
print(f'Specificity:{round(specificity_score(y_test,pred),3)}')
print('================================================================================================')
y_proba_knn = knn.predict_proba(X_test)[::,-1]
"""
===================
Confusion Matrix knn
====================
"""
matrix = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=knn.classes_)
disp.plot()
plt.title('KNN Confusion Matrix')
plt.show()

del knn


param_grid = {'weights':['uniform', 'distance'],
              'algorithm':['ball_tree', 'kd_tree', 'brute']}
KNN_GRID = GridSearchCV(KNeighborsClassifier(n_jobs=n_jobs),param_grid,verbose=1,
                      n_jobs=n_jobs,scoring='accuracy',cv=tscv)
KNN_GRID.fit(X,y.ravel())
knn_best2 = KNN_GRID.best_params_
print(knn_best2)

param_grid = {'p':[1,2,3]}
KNN_GRID = GridSearchCV(KNeighborsClassifier(n_jobs=n_jobs,weights=knn_best2['weights'],
                                             algorithm=knn_best2['algorithm']),param_grid,verbose=1,
                      n_jobs=n_jobs,scoring='accuracy',cv=tscv)
KNN_GRID.fit(X,y.ravel())
knn_best3 = KNN_GRID.best_params_
print(knn_best3)

param_grid = {'n_neighbors':np.arange(5,20,1)}
KNN_GRID = GridSearchCV(KNeighborsClassifier(n_jobs=n_jobs,weights=knn_best2['weights'],
                                             algorithm=knn_best2['algorithm'],p=knn_best3['p']),param_grid,verbose=1,
                      n_jobs=n_jobs,scoring='accuracy',cv=tscv)
KNN_GRID.fit(X,y.ravel())
knn_best = KNN_GRID.best_params_
print(knn_best)

knn_grid = KNeighborsClassifier(n_jobs=n_jobs,n_neighbors=knn_best['n_neighbors']
                           ,weights=knn_best2['weights'],algorithm=knn_best2['algorithm'],p=knn_best3['p'])
knn_grid.fit(X_train,y_train)
pred = knn_grid.predict(X_test)

y_proba_knn_grid = knn_grid.predict_proba(X_test)[::,-1]

print('========================')
print('After Grid Search KNN')
print(f'ACCURACY: {round(accuracy_score(y_test,pred),3)}')
print(f'F1: {round(f1_score(y_test,pred),3)}')
print(f'Precision: {round(precision_score(y_test,pred),3)}')
print(f'Recall: {round(recall_score(y_test,pred),3)}')
print(f'Specificity:{round(specificity_score(y_test,pred),3)}')
print('================================================================================================')

"""
===================
Confusion Matrix knn
====================
"""
matrix = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=knn_grid.classes_)
disp.plot()
plt.title('Grid Searched KNN Matrix')
plt.show()
del knn_grid


"""
====================
SVM
====================
"""
print('================================================================================================')
print('SVM')
svm = SVC(verbose=0,probability=True)
svm.fit(X_train,y_train)
pred = svm.predict(X_test)
print(f'ACCURACY: {round(accuracy_score(y_test,pred),3)}')
print(f'F1: {round(f1_score(y_test,pred),3)}')
print(f'Precision: {round(precision_score(y_test,pred),3)}')
print(f'Recall: {round(recall_score(y_test,pred),3)}')
print(f'Specificity:{round(specificity_score(y_test,pred),3)}')
print('================================================================================================')
y_proba_svm = svm.predict_proba(X_test)[::,-1]

"""
================
Confusion Matrix
================
"""
matrix = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=svm.classes_)
disp.plot()
plt.title('SVM Confusion Matrix')
plt.show()
del svm

param_grid = {'kernel':["linear", "poly", "rbf", "sigmoid"]}
SVM_GRID = GridSearchCV(SVC(),param_grid,verbose=0,
                      n_jobs=n_jobs,scoring='accuracy',cv=tscv)
SVM_GRID.fit(X,y.ravel())
svm_best = SVM_GRID.best_params_
print(svm_best)

best_svm = SVC(kernel=svm_best['kernel'],probability=True)
best_svm.fit(X_train,y_train)
pred = best_svm.predict(X_test)
print('========================')
print('After Grid Search KNN')
print(f'ACCURACY: {round(accuracy_score(y_test,pred),3)}')
print(f'F1: {round(f1_score(y_test,pred),3)}')
print(f'Precision: {round(precision_score(y_test,pred),3)}')
print(f'Recall: {round(recall_score(y_test,pred),3)}')
print(f'Specificity:{round(specificity_score(y_test,pred),3)}')
print('================================================================================================')
y_proba_svm_grid = best_svm.predict_proba(X_test)[::,-1]

"""
================================
Confusion Matrix SVM Grid Search
================================
"""
matrix = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=best_svm.classes_)
disp.plot()
plt.title('Grid-Searched SVM Confusion Matrix')
plt.show()
print('================================================================================================')
del best_svm
print('================================================================================================')

"""
====================
Naive Bayes
====================
"""
print('================================================================================================')
print('Gaussian NAIVE BAYES')
nb = GaussianNB()
nb.fit(X_train,y_train)
pred = nb.predict(X_test)
y_proba_nb = nb.predict_proba(X_test)[::,-1]
"""
===================
Confusion Matrix nb
===================
"""
matrix = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=nb.classes_)
disp.plot()
plt.title('Naive Bayes Confusion Matrix')
plt.show()
print('================================================')
print(f'ACCURACY: {round(accuracy_score(y_test,pred),3)}')
print(f'F1: {round(f1_score(y_test,pred),3)}')
print(f'Precision: {round(precision_score(y_test,pred),3)}')
print(f'Recall: {round(recall_score(y_test,pred),3)}')
print(f'Specificity:{round(specificity_score(y_test,pred),3)}')
print('================================================================================================')
del nb


"""
====================
Random Forest 
====================
"""
print('================================================================================================')
rf = RandomForestClassifier(n_jobs=n_jobs)
rf.fit(X_train,y_train)
pred = rf.predict(X_test)
print('Random Forest')
print(f'ACCURACY: {round(accuracy_score(y_test,pred),3)}')
print(f'F1: {round(f1_score(y_test,pred),3)}')
print(f'Precision: {round(precision_score(y_test,pred),3)}')
print(f'Recall: {round(recall_score(y_test,pred),3)}')
print(f'Specificity:{round(specificity_score(y_test,pred),3)}')
print('================================================================================================')
y_proba_rf = rf.predict_proba(X_test)[::,-1]
"""
===================
Confusion Matrix rf
===================
"""
matrix = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=rf.classes_)
disp.plot()
plt.title('Random Forest Confusion Matrix')
plt.show()
del rf

param_grid = {'criterion':['gini', 'entropy','log_loss'],
              'max_features':['auto', 'sqrt', 'log2'],
              'min_weight_fraction_leaf':range,
              'min_samples_leaf':np.arange(0,5,1),
              'min_impurity_decrease':range,
              'min_samples_split':np.arange(2,3,1)}

rf_grid = GridSearchCV(RandomForestClassifier(n_jobs=n_jobs,max_depth=10),
                       cv=tscv,param_grid=param_grid,n_jobs=n_jobs,verbose=1)
range = [0.0,0.01,0.02]
rf_grid.fit(X,y.ravel())
rf_grid_result = rf_grid.best_params_
print(rf_grid.best_params_)
rf_grid = RandomForestClassifier(max_depth=10,
                        criterion=rf_grid_result['criterion'],
                        max_features=rf_grid_result['max_features'],
    min_weight_fraction_leaf=rf_grid_result['min_weight_fraction_leaf'],
    min_samples_leaf=rf_grid_result['min_samples_leaf'],min_impurity_decrease=rf_grid_result['min_impurity_decrease'],
min_samples_split=rf_grid_result['min_samples_split'])

rf_grid.fit(X_train,y_train)
pred = rf_grid.predict(X_test)
print('==================================')
print('Grid Search Random Forest')
print(f'ACCURACY: {round(accuracy_score(y_test,pred),3)}')
print(f'F1: {round(f1_score(y_test,pred),3)}')
print(f'Precision: {round(precision_score(y_test,pred),3)}')
print(f'Recall: {round(recall_score(y_test,pred),3)}')
print(f'Specificity:{round(specificity_score(y_test,pred),3)}')
print('================================================================================================')
y_proba_rf_grid = rf_grid.predict_proba(X_test)[::,-1]
"""
================
Confusion Matrix after grid rf
================
"""
matrix = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=rf_grid.classes_)
disp.plot()
plt.title('Grid Search Random Forest Confusion Matrix')
plt.show()
del rf_grid


"""
======================
Bagging Random Forest
======================
"""
print('================================================================================================')
bc = BaggingClassifier(RandomForestClassifier(n_jobs=n_jobs),n_jobs=n_jobs)
bc.fit(X_train,y_train)
pred = bc.predict(X_test)
print('Bagging Random Forest')
print(f'ACCURACY: {round(accuracy_score(y_test,pred),3)}')
print(f'F1: {round(f1_score(y_test,pred),3)}')
print(f'Precision: {round(precision_score(y_test,pred),3)}')
print(f'Recall: {round(recall_score(y_test,pred),3)}')
print(f'Specificity:{round(specificity_score(y_test,pred),3)}')
print('================================================================================================')
y_proba_bag = bc.predict_proba(X_test)[::,-1]

"""
========================
Confusion Matrix bagging
========================
"""
matrix = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=bc.classes_)
disp.plot()
plt.title('Bagging Random Forest Confusion Matrix')
plt.show()
print('================================================================================================')
del bc

"""
====================
Multi Layered Perceptron
====================
"""
print('================================================================================================')

mlp = MLPClassifier(verbose=False,activation='identity',shuffle=True)
mlp.fit(X_train,y_train)
pred = mlp.predict(X_test)
print('========================')
print('Multilayer Perceptron')
print(f'ACCURACY: {round(accuracy_score(y_test,pred),3)}')
print(f'F1: {round(f1_score(y_test,pred),3)}')
print(f'Precision: {round(precision_score(y_test,pred),3)}')
print(f'Recall: {round(recall_score(y_test,pred),3)}')
print(f'Specificity:{round(specificity_score(y_test,pred),3)}')
print('================================================================================================')
y_proba_mlp = mlp.predict_proba(X_test)[::,-1]

"""
================
Confusion Matrix MLP
================
"""
matrix = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=mlp.classes_)
disp.plot()
plt.title('Multi Layered Perceptron Confusion Matrix')
plt.show()

"""
===========
ROC CURVES
===========
"""
plt.figure(figsize=(12,8))
"Log reg"
'y_proba_lg = lg.predict_proba(X_test)[::,-1]'
lg_fpr,lg_tpr,_ = metrics.roc_curve(y_test,y_proba_lg)
auc_lg = metrics.roc_auc_score(y_test,y_proba_lg)
plt.plot(lg_fpr,lg_tpr, label = f'Log Reg AUC = {auc_lg:.3f}')
plt.plot(lg_fpr,lg_fpr)

"Decision Tree"
'y_proba_dt = dt.predict_proba(X_test)[::,-1]'
dt_fpr,dt_tpr,_ = metrics.roc_curve(y_test,y_proba_dt)
auc_dt = metrics.roc_auc_score(y_test,y_proba_dt)
plt.plot(dt_fpr,dt_tpr, label = f'Decision Tree AUC = {auc_dt:.3f}')

"Decision Tree pre pruned"
dt_pre_fpr,dt_pre_tpr,_ = metrics.roc_curve(y_test,y_proba_dt_pre)
auc_dt_pre = metrics.roc_auc_score(y_test,y_proba_dt_pre)
plt.plot(dt_pre_fpr,dt_pre_tpr, label = f'Pre-Pruned-Decision Tree AUC = {auc_dt_pre:.3f}')

"Post pruned Decision Tree"
'y_proba_dt_post = dt_post.predict_proba(X_test)[::,-1]'
dt_post_fpr,dt_post_tpr,_ = metrics.roc_curve(y_test,y_proba_dt_post)
auc_dt_post = metrics.roc_auc_score(y_test,y_proba_dt_post)
plt.plot(dt_post_fpr,dt_post_tpr, label = f'Post-Pruned-Decision Tree AUC = {auc_dt_post:.3f}')

"KNN"
'y_proba_knn = knn.predict_proba(X_test)[::,-1]'
knn_fpr,knn_tpr,_ = metrics.roc_curve(y_test,y_proba_knn)
auc_knn = metrics.roc_auc_score(y_test,y_proba_knn)
plt.plot(knn_fpr,knn_tpr, label = f'KNN AUC = {auc_knn:.3f}')

'KNN Grid'
'y_proba_knn_grid = knn_grid.predict_proba(X_test)[::,-1]'
svm_fpr,svm_tpr,_ = metrics.roc_curve(y_test,y_proba_knn_grid)
auc_svm = metrics.roc_auc_score(y_test,y_proba_knn_grid)
plt.plot(svm_fpr,svm_tpr, label = f'KNN GRID AUC = {auc_svm:.3f}')


"Naive Bayes"
'y_proba_nb = nb.predict_proba(X_test)[::,-1]'
nb_fpr,nb_tpr,_ = metrics.roc_curve(y_test,y_proba_nb)
auc_nb = metrics.roc_auc_score(y_test,y_proba_nb)
plt.plot(nb_fpr,nb_tpr, label = f'Naive Bayes AUC = {auc_nb:.3f}')


"SVM"
'y_proba_svm = svm.predict_proba(X_test)[::,-1]'
svm_fpr,svm_tpr,_ = metrics.roc_curve(y_test,y_proba_svm)
auc_svm = metrics.roc_auc_score(y_test,y_proba_svm)
plt.plot(svm_fpr,svm_tpr, label = f'SVM AUC = {auc_svm:.3f}')

"SVM Grid"
'y_proba_svm_grid = best_svm.predict_proba(X_test)[::,-1]'
svm_fpr,svm_tpr,_ = metrics.roc_curve(y_test,y_proba_svm_grid)
auc_svm = metrics.roc_auc_score(y_test,y_proba_svm_grid)
plt.plot(svm_fpr,svm_tpr, label = f'Grid search SVM AUC = {auc_svm:.3f}')

"Random Forest"
'y_proba_rf = rf.predict_proba(X_test)[::,-1]'
rf_fpr,rf_tpr,_ = metrics.roc_curve(y_test,y_proba_rf)
auc_rf = metrics.roc_auc_score(y_test,y_proba_rf)
plt.plot(rf_fpr,rf_tpr, label = f'Random Forest AUC = {auc_rf:.3f}')


'Random Forest Grid search'
'y_proba_rf_grid = rf_grid.predict_proba(X_test)[::,-1]'
rf_fpr,rf_tpr,_ = metrics.roc_curve(y_test,y_proba_rf_grid)
auc_rf = metrics.roc_auc_score(y_test,y_proba_rf_grid)
plt.plot(rf_fpr,rf_tpr, label = f' Grid Search Random Forest AUC = {auc_rf:.3f}')


'Bagging Classifier'
'y_proba_bag = bc.predict_proba(X_test)[::,-1]'
rf_fpr,rf_tpr,_ = metrics.roc_curve(y_test,y_proba_bag)
auc_rf = metrics.roc_auc_score(y_test,y_proba_bag)
plt.plot(rf_fpr,rf_tpr, label = f'Bagging AUC = {auc_rf:.3f}')


"MLP"
'y_proba_mlp = mlp.predict_proba(X_test)[::,-1]'
mlp_fpr,mlp_tpr,_ = metrics.roc_curve(y_test,y_proba_mlp)
auc_mlp = metrics.roc_auc_score(y_test,y_proba_mlp)
plt.plot(mlp_fpr,mlp_tpr, label = f'MLP AUC = {auc_mlp:.3f}')

plt.grid()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('AUC CURVE')
plt.legend(bbox_to_anchor=(1.04,0.5), loc='center left')
plt.tight_layout()
plt.show()

end = time.time()
print('================================================================================================')
print(f'Elapsed Time: {end - start}')