import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from scipy import stats
from sklearn.preprocessing import StandardScaler,LabelEncoder,Normalizer,normalize
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.model_selection import train_test_split,TimeSeriesSplit,GridSearchCV
from sklearn.metrics import (mean_squared_error,
                             r2_score, accuracy_score, f1_score,
                             ConfusionMatrixDisplay, confusion_matrix, recall_score, precision_score, roc_auc_score,roc_curve,)
from prettytable import PrettyTable
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from imblearn.metrics import specificity_score
from imblearn.over_sampling import ADASYN
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
import warnings
import time

"""
======================
Micah Harlan
======================
"""

gc.enable()
'Starting Time'
start = time.time()
warnings.filterwarnings("ignore")
n_jobs=-1

read = 'pollution_2000_2023.csv'
df = pd.read_csv(read)
df.drop(columns=['Unnamed: 0'],inplace=True)

"""
======================
Number of Observations
======================
"""

air_qual = ['Good','Moderate','Unhealthy for SG','Unhealthy','Very Unhealthy']
ranges = [(0,50),(51,100),(101,150),(151,200),(201,300)]

temp = df['O3 AQI']
O3_AQI_class = pd.DataFrame()
removed_states = ['Alabama','Alabama', 'Arizona', 'Arkansas', 'California', 'Colorado'
    ,'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Kansas', 'Louisiana','Michigan',
                  'Minnesota', 'Missouri','Nevada', 'New Mexico',
                  'North Dakota','Ohio','Oklahoma', 'Oregon', 'South Dakota', 'Tennessee',
                  'Texas', 'Utah','Washington', 'Wisconsin','Wyoming','Mississippi','Iowa','Kentucky','Alaska',]

for s in removed_states:
    df = df[(df['State'] != s)]


print(df['NO2 Mean'])
print(len(df))

"PPM = Parts Per Million"
"PPB = Parts Per Billion"
units = {'O3':'PPM','NO2':'PPB','S02':'PPB','CO2':'PPM'}

"""
===============
Adding AQI Labels from these Sources Below.
"NITROGEN: https://document.airnow.gov/air-quality-guide-for-nitrogen-dioxide.pdf"
"OZONE: https://document.airnow.gov/air-quality-guide-for-ozone.pdf"
"Nice info https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf"
"https://document.airnow.gov"
===============
"""

df = df.dropna()
df.drop_duplicates(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Day'] = df['Date'].dt.day
print(df)

"""
======================
Adding Seasons
======================
"""
df['Season'] = np.select([df['Month'].between(3,5),
                        df['Month'].between(6,8),
                        df['Month'].between(9,11),
                        df['Month'] == 12,
                        df['Month'].between(1,2)],
                         ['Spring','Summer','Fall','Winter','Winter']
                         )
numeric = df.select_dtypes(include=np.number)
'''"""
=========================
Removing Outliers Z-scores
=========================
"""
threshold = 2.5
numeric = df.select_dtypes(include=np.number)
z_scores = stats.zscore(numeric)

outliers = numeric[abs(z_scores) > threshold]
outliers.dropna(axis=0,how='all',inplace=True)
outliers.dropna(axis=1,how='all',inplace=True)
print(outliers.describe)
'''
"""
==========================================
Highest AQI Value is the actual AQI value.
==========================================
"""

"New Feature Name: Chosen AQI"
"AQI value is chosen from the max AQI value of the 3."
df['AQI'] = df[['O3 AQI','CO AQI','SO2 AQI','NO2 AQI']].max(axis=1)
#df = df[(df['AQI'] > 20)]
df['Y'] = df['AQI'].shift(-1)
df.drop(df.tail(1).index,inplace=True)

"""
======================
Adding the AQI labels
======================
"""
df['NO2_AQI_label'] = np.select([df['NO2 AQI'].between(0,50),
                                 df['NO2 AQI'].between(51,100),
                                 df['NO2 AQI'].between(101,150),
                                 df['NO2 AQI'].between(151,200),
                                 df['NO2 AQI'].between(201,300)]
                                ,air_qual)

df['O3_AQI_label'] = np.select([df['O3 AQI'].between(0,50),
                                 df['O3 AQI'].between(51,100),
                                 df['O3 AQI'].between(101,150),
                                 df['O3 AQI'].between(151,200),
                                 df['O3 AQI'].between(201,300)]
                                ,air_qual)

df['CO_AQI_label'] = np.select([df['CO AQI'].between(0,50),
                                 df['CO AQI'].between(51,100),
                                 df['CO AQI'].between(101,150),
                                 df['CO AQI'].between(151,200),
                                 df['CO AQI'].between(201,300)]
                                ,air_qual)

df['SO2_AQI_label'] = np.select([df['SO2 AQI'].between(0,50),
                                 df['SO2 AQI'].between(51,100),
                                 df['SO2 AQI'].between(101,150),
                                 df['SO2 AQI'].between(151,200),
                                 df['SO2 AQI'].between(201,300)]
                                ,air_qual)

df['classification'] = np.select([df['Y'].between(0,50),
                                 df['Y'].between(51,100),
                                 df['Y'].between(101,150),
                                 df['Y'].between(151,200),
                                 df['Y'].between(201,300)]
                                ,air_qual)
"""
Setting Date
"""
shuffle = False
df = df[(df['Date'] >= '2017-01-01')]
df['days_since_start'] = (df['Date'] - pd.to_datetime('2017-01-01')).dt.days
df.reset_index(inplace=True,drop=True)
df.to_csv('classification.csv',index=False)
print(len(df))

"""
========================
Dimensionality Reduction
========================
"""

"""
==================
Low Variance Filter
==================
"""
normalize = normalize(numeric)
numeric_normalized = pd.DataFrame(normalize)
var = numeric_normalized.var()
var_normalized = numeric_normalized.var()*100/np.max(var)
threshold = 1e-3
low_variance = []
numeric_col = numeric.columns
for i in range(len(numeric_col)):
    if var_normalized[i] < threshold:
        low_variance.append(numeric.columns[i])
print(f'Low Variance Filtered Features {low_variance}')

"""
=====
Data Transformation
=====
"""
X = df.drop(columns=['classification','Address','Date'])
le = LabelEncoder()
le.fit(X['NO2_AQI_label'])
X['NO2_AQI_label'] = le.fit_transform(X['NO2_AQI_label'])
le.fit(X['O3_AQI_label'])
X['O3_AQI_label'] = le.fit_transform(X['O3_AQI_label'])
le.fit(X['CO_AQI_label'])
X['CO_AQI_label'] = le.fit_transform(X['CO_AQI_label'])
le.fit(X['SO2_AQI_label'])
X['SO2_AQI_label'] = le.fit_transform(X['SO2_AQI_label'])
le.fit(df['classification'])
y_class = le.fit_transform(df['classification'])

norm = Normalizer()
norm.fit(X['Month'].to_numpy().reshape(len(X['Month']),-1))
X['Month'] = norm.fit_transform(X['Month'].to_numpy().reshape(len(X['Month']),-1))

norm.fit(X['Day'].to_numpy().reshape(len(X['Day']),-1))
X['Day'] = norm.fit_transform(X['Day'].to_numpy().reshape(len(X['Day']),-1))

norm.fit(X['Year'].to_numpy().reshape(len(X['Year']),-1))
X['Year'] = norm.fit_transform(X['Year'].to_numpy().reshape(len(X['Year']),-1))

numerical = ['O3 1st Max Hour',
        'CO 1st Max Hour',
        'SO2 1st Max Hour', 'NO2 Mean',
       'NO2 1st Max Value', 'NO2 1st Max Hour',
             'O3 Mean', 'O3 1st Max Value', 'CO Mean','CO 1st Max Value', 'SO2 Mean',
             'SO2 1st Max Value','days_since_start']
std1 = StandardScaler()
for s in numerical:
    std1.fit(X[s].to_numpy().reshape(len(X[s]),-1))
    X[s] = std1.fit_transform(X[s].to_numpy().reshape(len(X[s]),-1))
std = StandardScaler()

std.fit(X['Y'].to_numpy().reshape(len(X['Y']),-1))
X['Y'] = std.fit_transform(X['Y'].to_numpy().reshape(len(X['Y']),-1))

y = X['Y']

numerical.append('AQI')
numerical.append('CO AQI')
numerical.append('SO2 AQI')
numerical.append('O3 AQI')
numerical.append('NO2 AQI')

std3 = StandardScaler()
std3.fit(X['AQI'].to_numpy().reshape(len(X['AQI']),-1))
X['AQI'] = std3.fit_transform(X['AQI'].to_numpy().reshape(len(X['AQI']),-1))

co = StandardScaler()
co.fit(X['CO AQI'].to_numpy().reshape(len(X['CO AQI']),-1))
X['CO AQI'] =co.fit_transform(X['CO AQI'].to_numpy().reshape(len(X['CO AQI']),-1))

so2 = StandardScaler()
so2.fit(X['SO2 AQI'].to_numpy().reshape(len(X['SO2 AQI']),-1))
X['SO2 AQI'] = so2.fit_transform(X['SO2 AQI'].to_numpy().reshape(len(X['SO2 AQI']),-1))

o3 = StandardScaler()
o3.fit(X['O3 AQI'].to_numpy().reshape(len(X['O3 AQI']),-1))
X['O3 AQI'] = o3.fit_transform(X['O3 AQI'].to_numpy().reshape(len(X['O3 AQI']),-1))

no2 = StandardScaler()
no2.fit(X['NO2 AQI'].to_numpy().reshape(len(X['NO2 AQI']),-1))
X['NO2 AQI'] = no2.fit_transform(X['NO2 AQI'].to_numpy().reshape(len(X['NO2 AQI']),-1))

X.drop(columns=['Y','County','City'],inplace=True,axis=1)
X = pd.get_dummies(X,drop_first=True,dtype='int')
X = sm.add_constant(X)

"""
========================================
Features Pearson Correrlation
========================================
"""

temp = X[numerical]
features = temp.keys()
pearson_corr = df[features]
plt.figure(figsize=(10,10))
correlation_matrix = pearson_corr.corr(method='pearson')
mask = np.triu(np.ones_like(correlation_matrix,dtype=bool),k=1)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",mask=mask)
plt.title('Pearson Correlation Coefficients Heatmap Matrix Heatmap')
plt.tight_layout()
plt.show()

"""
========================================
Features Covariance Matrix
========================================
"""
temp = df[numerical]
plt.figure(figsize=(12,12))
mask = np.triu(np.ones_like(temp.cov(),dtype=bool),k=2)
sns.heatmap(temp.cov(), annot=True, cmap='coolwarm', fmt=".2f",mask=mask)
plt.title('Covariance Matrix Heatmap')
plt.tight_layout()
plt.show()



"""
=========
Dropping low variance features
==========
"""
X.drop(low_variance, inplace=True,axis=1)

#%%
"""
===============
VIF Analysis
===============
"""
#df.drop('classification',inplace=True,axis=1)
under_ten = False
removed =[]

while not under_ten:
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    if vif_data['VIF'].max() <= 10:
        under_ten = True
        break
    t = vif_data.loc[vif_data['VIF'] == vif_data['VIF'].max()]
    X.drop(t['feature'].item(),inplace=True,axis=1)
    removed.append(t['feature'].item())
removed.append('CO_AQI_label')
X.drop('CO_AQI_label',inplace=True,axis=1)
vif_data1 = pd.DataFrame()
vif_data1['feature'] = X.columns
vif_data1['VIF'] = [variance_inflation_factor(X.values,i) for i in range(len(X.columns))]
print(vif_data1)
print(f'FEATURES REMOVED WITH MULTICOLINEARITY: {len(removed)}')
print(f'REMOVED: {removed}')
X = sm.add_constant(X)
copy_of_x = X.copy()
vif_dropped = removed

"""
======================
Random Forest Analysis
======================
"""
rf = RandomForestRegressor(max_depth=10,n_jobs=n_jobs)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2 ,shuffle=shuffle)
rf.fit(X_train,y_train)
importances = rf.feature_importances_
features = X.columns
indices = np.argsort(importances)

"""
=====================================
Feature importance threshold dropping
=====================================
"""
threshold = 0.025
dropped = []
kept = []
importance = []
i = 0
print(f'\n')
for ind in indices:
    if importances[ind] <= threshold:
        dropped.append(features[ind])
        X.drop(columns=features[ind],inplace=True,axis=1)
print(f'\n')
print(f'RF ANALYSIS FEATURES DROPPED: {dropped}')

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2 ,shuffle=shuffle)

rf.fit(X_train,y_train)
rand_kept = X.columns
importances = rf.feature_importances_
features = X.columns
indices = np.argsort(importances)
plt.title(f'Feature Importance After {len(dropped)} dropped')
plt.barh(range(len(indices)),importances[indices],color='b',align='center')
plt.yticks(range(len(indices)),[features[i] for i in indices])
plt.xlabel('Relative Importance.')
plt.tight_layout()
plt.show()


"""
============================
PRINCIPAL COMPONENT ANALYSIS
============================
"""
"""
MAKE SURE TO PLOT 90% of explained variance line
"""
X = copy_of_x.copy()
pca = PCA(svd_solver='arpack')
pca.fit(X)
X_pca = pca.transform(X)
print(f'Explained Variance Ratio {pca.explained_variance_ratio_}')
s = 0
a = 0
for i in pca.explained_variance_ratio_:
    s += i
    a += 1
    if s >= .90:
        break
n_features = len(X.columns)

plt.figure()
plt.plot(np.arange(1,len(np.cumsum(pca.explained_variance_ratio_))
+1,1),np.cumsum(pca.explained_variance_ratio_),label='Variance Ratio')
plt.xticks(np.arange(1,len(np.cumsum(pca.explained_variance_ratio_))+1,10))
plt.xlabel('N Components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA Exlplained Variance vs N components')
plt.legend()
plt.grid()
plt.show()

print(f'RF DROPPED: {dropped}')
print(f'LOW VARIANCE DROPPED: {low_variance}')
print(f'VIF DROPPED: {vif_dropped}')
dropped_features = list(set(dropped).intersection(low_variance))

kept_features = ['AQI','days_since_start','O3_AQI_label','NO2 Mean']
X = copy_of_x.copy()
X.drop(columns=[col for col in X.keys() if col not in rand_kept],inplace=True)
print(X.columns)

'''
"""
========================================
Selected Features Pearson Correrlation
========================================
"""
temp = X.copy()
temp.drop(['O3_AQI_label'],inplace=True,axis=1)
features = temp.keys()
pearson_corr = df[features]
plt.figure(figsize=(6,6))
correlation_matrix = pearson_corr.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation Coefficients Heatmap Matrix Heatmap')
plt.tight_layout()
plt.show()

"""
========================================
Selected Features Covariance Matrix
========================================
"""
plt.figure(figsize=(6,6))
sns.heatmap(temp.cov(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Covariance Matrix Heatmap')
plt.tight_layout()
plt.show()
X = df[X.keys()]
print(f' {X.keys()}')
print(X)
X['Y'] = df['Y']
'''
#X.to_csv('cleaned_aqi.csv',index=False)
'''
print('===============================================')
print('***********************************************')
print('****************** PHASE II *******************')
print('************ Regression Analysis **************')
print('***********************************************')
print('===============================================')
print('')

df = pd.read_csv('cleaned_aqi.csv')
y = df['Y']
X = df.drop('Y',axis=1)
shuffle = False
y_std = StandardScaler()
y_std.fit(y.to_numpy().reshape(len(y),-1))
y = y_std.fit_transform(y.to_numpy().reshape(len(y),-1))
std = StandardScaler()
for k in X.keys():
    if k == 'O3_AQI_label':
        continue
    std.fit(X[k].to_numpy().reshape(len(X[k]), -1))
    X[k] = std.fit_transform(X[k].to_numpy().reshape(len(X[k]), -1))
le = LabelEncoder()
le.fit(X['O3_AQI_label'])
X['O3_AQI_label'] = le.fit_transform(X['O3_AQI_label'])
X = sm.add_constant(X)
tscv = TimeSeriesSplit(n_splits=12)
copy_x = X.copy()
X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=shuffle,test_size=.2 )

"""
===============================
Backwards Stepwise
===============================
"""
all_zero = False
removed =[]
model = sm.OLS(y_train, X_train).fit()
print(model.summary())
while not all_zero:
    ar = model.pvalues.reset_index()
    "Checks for each one equals 0"
    if round(ar[0].sum(),3) == 0.000:
        all_zero = True
        break
    t = ar.loc[ar[0] == ar[0].max()]
    remove = t['index'].item()
    removed.append(remove)
    X_test.drop(remove,inplace=True,axis=1)
    X_train.drop(remove, inplace=True,axis=1)
    model = sm.OLS(y_train, X_train).fit()
print(f'Stepwise FEATURES REMOVED: {len(removed)}')
step_wise = removed
print(model.summary())

"""
Plotting Predicted Values
"""
plt.figure(figsize=(10,4))
fig,axs = plt.subplots(2,1)
predictions = model.predict(X_test)
predictions_rev = y_std.inverse_transform(predictions.to_numpy().reshape(len(X_test),1))
actual = y_std.inverse_transform(y_test.reshape(len(X_test),1))
plot_pred_rev = pd.DataFrame(predictions_rev).rolling(window=50).mean()
conf = plot_pred_rev
actual_plot = pd.DataFrame(actual).rolling(window=50).mean()
sns.lineplot(ax=axs[0],x=np.arange(0,len(plot_pred_rev),1),y=plot_pred_rev.to_numpy().reshape(len(predictions_rev),),label='Predicted',color='blue')
sns.lineplot(ax=axs[0],x=np.arange(0,len(actual_plot),1),y=actual_plot.to_numpy().reshape(len(X_test),),alpha=.5,label='Actual',color='green')
axs[0].set_xlabel('N Observations')
axs[0].set_ylabel('Actual Value')
axs[0].set_title(f'Backwards Stepwise: Actual vs. Predicted Value RMSE: {round(mean_squared_error(actual,predictions_rev,squared=False),2)}')
test_predictions = model.predict(X_train)
test = y_std.inverse_transform(test_predictions.to_numpy().reshape(len(test_predictions),1))
actual2 = y_std.inverse_transform(y_train.reshape(len(y_train),1))
plot_pred_rev = pd.DataFrame(test).rolling(window=600).mean()
actual_plot = pd.DataFrame(actual2).rolling(window=600).mean()
sns.lineplot(ax=axs[1],x=np.arange(0,len(plot_pred_rev),1),y=plot_pred_rev.to_numpy().reshape(len(test),),label='Train Set Predicted',color='blue')
sns.lineplot(ax=axs[1],x=np.arange(0,len(actual_plot),1),y=actual_plot.to_numpy().reshape(len(X_train),),label='Train Actual',alpha=0.4,color='green')
axs[1].set_xlabel('N Observations')
axs[1].set_ylabel('Actual Value')
axs[1].set_title(f'Backwards Stepwise: Actual Train set vs. Predicted Value RMSE: {round(mean_squared_error(actual2,test,squared=False),2)}')
fig.tight_layout()
fig.show()

"""
=======================
Confidence Interval
=======================
"""
X_test2 = pd.DataFrame(X_test).rolling(window=25).mean()
plot_pred_rev = conf
confidence = model.get_prediction(X_test2).conf_int()
confidence = y_std.inverse_transform(confidence)
plt.figure(figsize=(10,4))

sns.lineplot(data=plot_pred_rev,
x=np.arange(len(plot_pred_rev)),y=plot_pred_rev.to_numpy().reshape(len(plot_pred_rev),),label='Predicted',linestyle='--')
plt.fill_between(
np.arange(len(plot_pred_rev)),confidence[:,0] , confidence[:,1],
color='g',label='Confidence Interval')
sns.set_theme(style="whitegrid")
plt.xlabel('# of Samples')
plt.ylabel('AQI')
plt.title('Rolling avg: Backwards Stepwise vs Confidence interval')
plt.legend()
plt.show()

"""
=========================================================
EXTRACTING R-squared, adjusted R-square, AIC, BIC and MSE + RMSE
=========================================================
"""
ols_r2 = round(model.rsquared,3)
ols_adj_r2 = round(model.rsquared_adj,3)
ols_AIC = round(model.aic,3)
ols_BIC = round(model.bic,3)
ols_MSE = round(mean_squared_error(actual,predictions_rev),2)

tab = ["Linear Regression", ols_r2, ols_adj_r2,ols_AIC,ols_BIC,ols_MSE]


X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=shuffle,test_size=.2 )
"""
====================================
RANDOM FOREST FOR FEATURE SELECTION
====================================
"""
X = copy_x.copy()
rf = RandomForestRegressor(max_depth=10)
rf.fit(X_train,y_train)
importances = rf.feature_importances_
features = X.columns
indices = np.argsort(importances)
rand_kept = X.columns
importances = rf.feature_importances_
plt.figure()
plt.title(f'Feature Importance')
plt.barh(range(len(indices)),importances[indices],color='b',align='center')
plt.yticks(range(len(indices)),[features[i] for i in indices])
plt.xlabel('Relative Importance.')
plt.tight_layout()
plt.show()

X.drop('const',inplace=True,axis=1)
X_train.drop('const',inplace=True,axis=1)
X_test.drop('const',inplace=True,axis=1)
rf.fit(X_train,y_train)
predictions = rf.predict(X_test)
predictions_rev = y_std.inverse_transform(predictions.reshape(len(X_test),1))
actual = y_std.inverse_transform(y_test.reshape(len(X_test),1))

"""
=========================================================
EXTRACTING R-squared, adjusted R-square, AIC, BIC and MSE + RMSE
=========================================================
"""
rf_MSE = round(mean_squared_error(actual,predictions_rev),2)
k = len(rf.get_params() ) + rf.n_estimators

rf_r2 = r2_score(actual, predictions_rev)
rf_adj_r2 = 1 - (1 - rf_r2) * ((len(X_test) - 1) / (len(X_test) - len(X_test.keys()) - 1))
rf_AIC = round(2 * k - 2 * np.log(rf_MSE),3)

rf_BIC = len(X_train) * np.log(rf_MSE) + k * np.log(len(X_train))

del k
table = PrettyTable()

# Define table columns
table.field_names = ['Model','R-squared', 'adjusted R-square', 'AIC', 'BIC','MSE']

# Add data to the table
table.add_row(["Random Forest", round(rf_r2,3), round(rf_adj_r2,3),round(rf_AIC,3),round(rf_BIC,3),rf_MSE])
table.add_row(tab)
print(table)

"""
plotting predicted values
"""

plt.figure(figsize=(10,4))
fig,axs = plt.subplots(2,1)
predictions = rf.predict(X_test)
predictions_rev = y_std.inverse_transform(predictions.reshape(len(X_test),1))
actual = y_std.inverse_transform(y_test.reshape(len(X_test),1))
plot_pred_rev = pd.DataFrame(predictions_rev).rolling(window=50).mean()
conf = plot_pred_rev
actual_plot = pd.DataFrame(actual).rolling(window=50).mean()
sns.lineplot(ax=axs[0],x=np.arange(0,len(plot_pred_rev),1),y=plot_pred_rev.to_numpy().reshape(len(predictions_rev),),label='Predicted',color='blue')
sns.lineplot(ax=axs[0],x=np.arange(0,len(actual_plot),1),y=actual_plot.to_numpy().reshape(len(X_test),),alpha=.5,label='Actual',color='green')
axs[0].set_xlabel('N Observations')
axs[0].set_ylabel('Actual Value')
axs[0].set_title(f'Random Forest: Actual vs. Predicted Value RMSE: {round(mean_squared_error(actual,predictions_rev,squared=False),2)}')
test_predictions = rf.predict(X_train)
test = y_std.inverse_transform(test_predictions.reshape(len(test_predictions),1))
actual2 = y_std.inverse_transform(y_train.reshape(len(y_train),1))
plot_pred_rev = pd.DataFrame(test).rolling(window=600).mean()
actual_plot = pd.DataFrame(actual2).rolling(window=600).mean()
sns.lineplot(ax=axs[1],x=np.arange(0,len(plot_pred_rev),1),y=plot_pred_rev.to_numpy().reshape(len(test),),label='Predicted',color='blue')
sns.lineplot(ax=axs[1],x=np.arange(0,len(actual_plot),1),y=actual_plot.to_numpy().reshape(len(X_train),),label='Actual',alpha=0.4,color='green')
axs[1].set_xlabel('N Observations')
axs[1].set_ylabel('Actual Value')
axs[1].set_title(f'Random Forest: Actual Train vs. Predicted Value RMSE: {round(mean_squared_error(actual2,test,squared=False),2)}')
fig.tight_layout()
fig.show()
plt.show()
del rf
del model
del X
del df'''
print('===============================================')
print('***********************************************')
print('****************** PHASE III ******************')
print('*************** Classification ****************')
print('***********************************************')
print('===============================================')
print('')


df = (pd.read_csv('classification.csv'))
df['classification'] = np.select([df['Y'].between(0,100),
                                 df['Y'].between(101,30000)]
                                ,['Healthy','Unhealthy'])

y = df['classification']
features_list = ['Year', 'NO2 Mean', 'NO2 1st Max Hour', 'AQI',
       'O3_AQI_label','days_since_start','SO2 Mean']
df.drop(columns=[col for col in df.keys() if col not in features_list],inplace=True)
X = df.copy()

"""
=====================
Transformations
=====================
"""
so2_mean_std = StandardScaler()
so2_mean_std.fit(X['SO2 Mean'].to_numpy().reshape(len(X['SO2 Mean']), -1))
X['SO2 Mean'] = so2.fit_transform(X['SO2 Mean'].to_numpy().reshape(len(X['SO2 Mean']),-1))


no2_mean_std = StandardScaler()
no2_mean_std.fit(X['NO2 Mean'].to_numpy().reshape(len(X['NO2 Mean']), -1))
X['NO2 Mean'] = no2_mean_std.fit_transform(X['NO2 Mean'].to_numpy().reshape(len(X['NO2 Mean']),-1))

no2_1st_max_hr_std = StandardScaler()
no2_1st_max_hr_std.fit(X['NO2 1st Max Hour'].to_numpy().reshape(len(X['NO2 Mean']), -1))
X['NO2 1st Max Hour'] = no2_1st_max_hr_std.fit_transform(X['NO2 1st Max Hour'].to_numpy().reshape(len(X['NO2 Mean']),-1))

aqi_std = StandardScaler()
aqi_std.fit(X['AQI'].to_numpy().reshape(len(X['NO2 Mean']), -1))
X['AQI'] = aqi_std.fit_transform(X['AQI'].to_numpy().reshape(len(X['NO2 Mean']),-1))

days_start_std = StandardScaler()
days_start_std.fit(X['days_since_start'].to_numpy().reshape(len(X['NO2 Mean']), -1))
X['days_since_start'] = days_start_std.fit_transform(X['days_since_start'].to_numpy().reshape(len(X['NO2 Mean']),-1))

year_std = StandardScaler()
year_std.fit(X['Year'].to_numpy().reshape(len(X['NO2 Mean']), -1))
X['Year'] = year_std.fit_transform(X['Year'].to_numpy().reshape(len(X['NO2 Mean']),-1))

le = LabelEncoder()
le.fit(X['O3_AQI_label'])
X['O3_AQI_label'] = le.fit_transform(X['O3_AQI_label'].to_numpy().reshape(len(X['NO2 Mean']),-1))

le_y = LabelEncoder()
le_y.fit(y)
y = le_y.fit_transform(y.to_numpy().reshape(len(X['NO2 Mean']),-1))

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

X['Year'] = year_std.inverse_transform(X['Year'].to_numpy().reshape(len(X['NO2 Mean']),-1))
X['days_since_start'] = days_start_std.inverse_transform(X['days_since_start'].to_numpy().reshape(len(X['NO2 Mean']),-1))

X.sort_values(['days_since_start','Year'],axis=0,inplace=True)
y = X['y_class']

X.drop(columns=['Year','y_class'],inplace=True,axis=1)
X['days_since_start'] = days_start_std.fit_transform(X['days_since_start'].to_numpy().reshape(len(X['NO2 Mean']),-1))

sns.countplot(x=y)
plt.title('Countplot of Target')
plt.tight_layout()
plt.show()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2 ,shuffle=shuffle)
tscv = TimeSeriesSplit(n_splits=12)

"""
====================
Logistic Regression
====================
"""
print('================================================================================================')

print('Logistic Regression')
lg = LogisticRegression(fit_intercept=True)
scores = cross_val_score(LogisticRegression(fit_intercept=True), X, y, cv=tscv,n_jobs=n_jobs)

lg.fit(X_train,y_train)
pred = lg.predict(X_test)

print(f'ACCURACY: {round(accuracy_score(y_test,pred),3)}')
print(f'F1: {round(f1_score(y_test,pred),3)}')
print(f'Precision: {round(precision_score(y_test,pred),3)}')
print(f'Recall: {round(recall_score(y_test,pred),3)}')
print(f'Specificity:{round(specificity_score(y_test,pred),3)}')
print(f'Cross Val Score:{round(scores.mean(),3)}')
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
plt.title('Logistic Regression Confusion Matrix')
plt.grid(False)
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
plt.grid(False)
plt.show()
del dt
vals = [0.0,0.01,0.02,0.03]
param_grid = {'criterion':['gini', 'entropy','log_loss'],
              'splitter':['best', 'random'],
              'max_features':['sqrt', 'log2'],
              'min_weight_fraction_leaf':vals,
              'min_samples_leaf':np.arange(0,.05,.01),
              'min_impurity_decrease':vals,
              'min_samples_split':np.arange(2,5,1)
              }

dt_grid = GridSearchCV(DecisionTreeClassifier(max_depth=10),
                       cv=tscv,param_grid=param_grid,n_jobs=n_jobs,verbose=1)

dt_grid.fit(X,y.ravel())
dt_grid_result = dt_grid.best_params_
print(dt_grid.best_params_)
dt_pre = DecisionTreeClassifier(max_depth=10,
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
print(f'Cross Val Score: {round(dt_grid.best_score_,3)}')
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
plt.grid(False)
plt.show()

param_grid = {'ccp_alpha':np.arange(0,.005,.0001)}
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
print(f'Cross Val Score: {round(dt_grid.best_score_,3)}')
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
plt.grid(False)

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
plt.grid(False)

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

param_grid = {'n_neighbors':np.arange(1,36,1)}
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
print(f'Cross Val Score: {round(KNN_GRID.best_score_,3)}')

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
plt.grid(False)

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
plt.grid(False)
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
print('After Grid Search SVM')
print(f'ACCURACY: {round(accuracy_score(y_test,pred),3)}')
print(f'F1: {round(f1_score(y_test,pred),3)}')
print(f'Precision: {round(precision_score(y_test,pred),3)}')
print(f'Recall: {round(recall_score(y_test,pred),3)}')
print(f'Specificity:{round(specificity_score(y_test,pred),3)}')
print(f'Cross Val Score: {round(SVM_GRID.best_score_,3)}')

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
plt.grid(False)
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
scores = cross_val_score(GaussianNB(), X, y, cv=tscv,n_jobs=n_jobs)


"""
===================
Confusion Matrix nb
===================
"""
matrix = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=nb.classes_)
disp.plot()
plt.grid(False)
plt.title('Naive Bayes Confusion Matrix')
plt.show()
print('================================================')
print(f'ACCURACY: {round(accuracy_score(y_test,pred),3)}')
print(f'F1: {round(f1_score(y_test,pred),3)}')
print(f'Precision: {round(precision_score(y_test,pred),3)}')
print(f'Recall: {round(recall_score(y_test,pred),3)}')
print(f'Specificity:{round(specificity_score(y_test,pred),3)}')
print(f'Cross Val Score:{round(scores.mean(),3)}')
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
plt.grid(False)
plt.show()
del rf

param_grid = {'criterion':['gini', 'entropy','log_loss'],
              'max_features':['auto', 'sqrt', 'log2'],
              'min_weight_fraction_leaf':vals,
              'min_samples_leaf':np.arange(0,5,1),
              'min_impurity_decrease':vals,
              'min_samples_split':np.arange(2,5,1),
              }

rf_gridcv = GridSearchCV(RandomForestClassifier(n_jobs=n_jobs,max_depth=10),
                       cv=tscv,param_grid=param_grid,n_jobs=n_jobs,verbose=1)
vals = [0.0,0.01,0.02]
rf_gridcv.fit(X,y.ravel())
rf_grid_result = rf_gridcv.best_params_
print(rf_gridcv.best_params_)
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
print(f'Cross Val Score: {round(rf_gridcv.best_score_,3)}')
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
plt.grid(False)
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
scores = cross_val_score(BaggingClassifier(RandomForestClassifier(n_jobs=n_jobs),n_jobs=n_jobs), X, y, cv=tscv,n_jobs=n_jobs)
print('Bagging Random Forest')
print(f'ACCURACY: {round(accuracy_score(y_test,pred),3)}')
print(f'F1: {round(f1_score(y_test,pred),3)}')
print(f'Precision: {round(precision_score(y_test,pred),3)}')
print(f'Recall: {round(recall_score(y_test,pred),3)}')
print(f'Specificity:{round(specificity_score(y_test,pred),3)}')
print(f'Cross Val Score: {round(scores.mean(),3)}')
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
plt.grid(False)
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
scores = cross_val_score(MLPClassifier(verbose=False,activation='identity',shuffle=True), X, y, cv=tscv,n_jobs=n_jobs)
print(f'Cross Val Score: {round(scores.mean(),3)}')
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
plt.grid(False)
plt.show()

"""
===========
ROC CURVES
===========
"""
plt.figure(figsize=(12,8))
"Log reg"
'y_proba_lg = lg.predict_proba(X_test)[::,-1]'
lg_fpr,lg_tpr,_ = roc_curve(y_test,y_proba_lg)
auc_lg = roc_auc_score(y_test,y_proba_lg)
plt.plot(lg_fpr,lg_tpr, label = f'Log Reg AUC = {auc_lg:.3f}')
plt.plot(lg_fpr,lg_fpr)

"Decision Tree"
'y_proba_dt = dt.predict_proba(X_test)[::,-1]'
dt_fpr,dt_tpr,_ = roc_curve(y_test,y_proba_dt)
auc_dt = roc_auc_score(y_test,y_proba_dt)
plt.plot(dt_fpr,dt_tpr, label = f'Decision Tree AUC = {auc_dt:.3f}')

"Decision Tree pre pruned"
dt_pre_fpr,dt_pre_tpr,_ = roc_curve(y_test,y_proba_dt_pre)
auc_dt_pre = roc_auc_score(y_test,y_proba_dt_pre)
plt.plot(dt_pre_fpr,dt_pre_tpr, label = f'Pre-Pruned-Decision Tree AUC = {auc_dt_pre:.3f}')

"Post pruned Decision Tree"
'y_proba_dt_post = dt_post.predict_proba(X_test)[::,-1]'
dt_post_fpr,dt_post_tpr,_ = roc_curve(y_test,y_proba_dt_post)
auc_dt_post = roc_auc_score(y_test,y_proba_dt_post)
plt.plot(dt_post_fpr,dt_post_tpr, label = f'Post-Pruned-Decision Tree AUC = {auc_dt_post:.3f}')

"KNN"
'y_proba_knn = knn.predict_proba(X_test)[::,-1]'
knn_fpr,knn_tpr,_ = roc_curve(y_test,y_proba_knn)
auc_knn = roc_auc_score(y_test,y_proba_knn)
plt.plot(knn_fpr,knn_tpr, label = f'KNN AUC = {auc_knn:.3f}')

'KNN Grid'
'y_proba_knn_grid = knn_grid.predict_proba(X_test)[::,-1]'
svm_fpr,svm_tpr,_ = roc_curve(y_test,y_proba_knn_grid)
auc_svm = roc_auc_score(y_test,y_proba_knn_grid)
plt.plot(svm_fpr,svm_tpr, label = f'KNN GRID AUC = {auc_svm:.3f}')

"Naive Bayes"
'y_proba_nb = nb.predict_proba(X_test)[::,-1]'
nb_fpr,nb_tpr,_ = roc_curve(y_test,y_proba_nb)
auc_nb = roc_auc_score(y_test,y_proba_nb)
plt.plot(nb_fpr,nb_tpr, label = f'Naive Bayes AUC = {auc_nb:.3f}')

"SVM"
'y_proba_svm = svm.predict_proba(X_test)[::,-1]'
svm_fpr,svm_tpr,_ = roc_curve(y_test,y_proba_svm)
auc_svm = roc_auc_score(y_test,y_proba_svm)
plt.plot(svm_fpr,svm_tpr, label = f'SVM AUC = {auc_svm:.3f}')

"SVM Grid"
'y_proba_svm_grid = best_svm.predict_proba(X_test)[::,-1]'
svm_fpr,svm_tpr,_ = roc_curve(y_test,y_proba_svm_grid)
auc_svm = roc_auc_score(y_test,y_proba_svm_grid)
plt.plot(svm_fpr,svm_tpr, label = f'Grid search SVM AUC = {auc_svm:.3f}')

"Random Forest"
'y_proba_rf = rf.predict_proba(X_test)[::,-1]'
rf_fpr,rf_tpr,_ = roc_curve(y_test,y_proba_rf)
auc_rf = roc_auc_score(y_test,y_proba_rf)
plt.plot(rf_fpr,rf_tpr, label = f'Random Forest AUC = {auc_rf:.3f}')

'Random Forest Grid search'
'y_proba_rf_grid = rf_grid.predict_proba(X_test)[::,-1]'
rf_fpr,rf_tpr,_ = roc_curve(y_test,y_proba_rf_grid)
auc_rf = roc_auc_score(y_test,y_proba_rf_grid)
plt.plot(rf_fpr,rf_tpr, label = f' Grid Search Random Forest AUC = {auc_rf:.3f}')

'Bagging Classifier'
'y_proba_bag = bc.predict_proba(X_test)[::,-1]'
rf_fpr,rf_tpr,_ = roc_curve(y_test,y_proba_bag)
auc_rf = roc_auc_score(y_test,y_proba_bag)
plt.plot(rf_fpr,rf_tpr, label = f'Bagging AUC = {auc_rf:.3f}')

"MLP"
'y_proba_mlp = mlp.predict_proba(X_test)[::,-1]'
mlp_fpr,mlp_tpr,_ = roc_curve(y_test,y_proba_mlp)
auc_mlp = roc_auc_score(y_test,y_proba_mlp)
plt.plot(mlp_fpr,mlp_tpr, label = f'MLP AUC = {auc_mlp:.3f}')

plt.grid()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('AUC CURVE')
plt.legend(bbox_to_anchor=(1.04,0.5), loc='center left')
plt.tight_layout()
plt.show()


print('===============================================')
print('***********************************************')
print('****************** PHASE III ******************')
print('********* Clustering and Rule Mining **********')
print('***********************************************')
print('===============================================')
print('')

read = 'pollution_2000_2023.csv'
df = pd.read_csv(read)
df.drop(columns=['Unnamed: 0'],inplace=True)

"""
======================
Number of Observations
======================
"""

air_qual = ['Good','Moderate','Unhealthy for SG','Unhealthy','Very Unhealthy']
ranges = [(0,50),(51,100),(101,150),(151,200),(201,300)]

temp = df['O3 AQI']
O3_AQI_class = pd.DataFrame()
removed_states = ['Alabama','Alabama', 'Arizona', 'Arkansas', 'California', 'Colorado'
    ,'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Kansas', 'Louisiana','Michigan',
                  'Minnesota', 'Missouri','Nevada', 'New Mexico',
                  'North Dakota','Ohio','Oklahoma', 'Oregon', 'South Dakota', 'Tennessee',
                  'Texas', 'Utah','Washington', 'Wisconsin','Wyoming','Mississippi','Iowa','Kentucky','Alaska',]

for s in removed_states:
    df = df[(df['State'] != s)]


"PPM = Parts Per Million"
"PPB = Parts Per Billion"
units = {'O3':'PPM','NO2':'PPB','S02':'PPB','CO2':'PPM'}


df = df.dropna()
df.drop_duplicates(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Day'] = df['Date'].dt.day

"""
======================
Adding Seasons
======================
"""
df['Season'] = np.select([df['Month'].between(3,5),
                        df['Month'].between(6,8),
                        df['Month'].between(9,11),
                        df['Month'] == 12,
                        df['Month'].between(1,2)],
                         ['Spring','Summer','Fall','Winter','Winter']
                         )

air_qual = ['NO2_Good','NO2_Moderate','NO2_Unhealthy for SG','NO2_Unhealthy','NO2_Very Unhealthy']
df['NO2_AQI_label'] = np.select([df['NO2 AQI'].between(0,50),
                                 df['NO2 AQI'].between(51,100),
                                 df['NO2 AQI'].between(101,150),
                                 df['NO2 AQI'].between(151,200),
                                 df['NO2 AQI'].between(201,300)]
                                ,air_qual)

air_qual = ['O3_Good','O3_Moderate','O3_Unhealthy for SG','O3_Unhealthy','O3_Very_Unhealthy']
df['O3_AQI_label'] = np.select([df['O3 AQI'].between(0,50),
                                 df['O3 AQI'].between(51,100),
                                 df['O3 AQI'].between(101,150),
                                 df['O3 AQI'].between(151,200),
                                 df['O3 AQI'].between(201,300)]
                                ,air_qual)

air_qual = ['CO_Good','CO_Moderate','CO_Unhealthy for SG','CO_Unhealthy','CO_Very Unhealthy']
df['CO_AQI_label'] = np.select([df['CO AQI'].between(0,50),
                                 df['CO AQI'].between(51,100),
                                 df['CO AQI'].between(101,150),
                                 df['CO AQI'].between(151,200),
                                 df['CO AQI'].between(201,300)]
                                ,air_qual)

air_qual = ['SO2_Good','SO2_Moderate','SO2_Unhealthy for SG','SO2_Unhealthy','SO2_Very Unhealthy']

df['SO2_AQI_label'] = np.select([df['SO2 AQI'].between(0,50),
                                 df['SO2 AQI'].between(51,100),
                                 df['SO2 AQI'].between(101,150),
                                 df['SO2 AQI'].between(151,200),
                                 df['SO2 AQI'].between(201,300)]
                                ,air_qual)


#%%
"""
Setting Date
"""
shuffle = False
df = df[(df['Date'] >= '2017-01-01')]
df['days_since_start'] = (df['Date'] - pd.to_datetime('2017-01-01')).dt.days
df.reset_index(inplace=True,drop=True)

"""
=====
Data Transformation
=====
"""
X = df.drop(columns=['Address','Date'])

norm = Normalizer()
norm.fit(X['Month'].to_numpy().reshape(len(X['Month']),-1))
X['Month'] = norm.fit_transform(X['Month'].to_numpy().reshape(len(X['Month']),-1))

norm.fit(X['Day'].to_numpy().reshape(len(X['Day']),-1))
X['Day'] = norm.fit_transform(X['Day'].to_numpy().reshape(len(X['Day']),-1))

norm.fit(X['Year'].to_numpy().reshape(len(X['Year']),-1))
X['Year'] = norm.fit_transform(X['Year'].to_numpy().reshape(len(X['Year']),-1))

numerical = ['O3 1st Max Hour',
        'CO 1st Max Hour',
        'SO2 1st Max Hour', 'NO2 Mean',
       'NO2 1st Max Value', 'NO2 1st Max Hour',
             'O3 Mean', 'O3 1st Max Value', 'CO Mean','CO 1st Max Value', 'SO2 Mean',
             'SO2 1st Max Value','days_since_start']


std1 = StandardScaler()
for s in numerical:
    std1.fit(X[s].to_numpy().reshape(len(X[s]),-1))
    X[s] = std1.fit_transform(X[s].to_numpy().reshape(len(X[s]),-1))

co = StandardScaler()
co.fit(X['CO AQI'].to_numpy().reshape(len(X['CO AQI']),-1))
X['CO AQI'] = co.fit_transform(X['CO AQI'].to_numpy().reshape(len(X['CO AQI']),-1))

so2 = StandardScaler()
so2.fit(X['SO2 AQI'].to_numpy().reshape(len(X['SO2 AQI']),-1))
X['SO2 AQI'] = so2.fit_transform(X['SO2 AQI'].to_numpy().reshape(len(X['SO2 AQI']),-1))

o3 = StandardScaler()
o3.fit(X['O3 AQI'].to_numpy().reshape(len(X['O3 AQI']),-1))
X['O3 AQI'] = o3.fit_transform(X['O3 AQI'].to_numpy().reshape(len(X['O3 AQI']),-1))

no2 = StandardScaler()
no2.fit(X['NO2 AQI'].to_numpy().reshape(len(X['NO2 AQI']),-1))
X['NO2 AQI'] = no2.fit_transform(X['NO2 AQI'].to_numpy().reshape(len(X['NO2 AQI']),-1))

apriori_labels = ['NO2_AQI_label',
                  'O3_AQI_label', 'CO_AQI_label','State','Season','SO2_AQI_label']
apriori_x = X[apriori_labels]

X.drop(columns =apriori_labels,inplace=True,axis=1)
X.drop(columns=['County','City'],inplace=True,axis=1)
copy_of_x = X.copy()


copy = X.copy()
X2 = X.copy()
X3 = X.copy()
pca = PCA()
principalComponents = pca.fit_transform(X)
PCA_components = pd.DataFrame(principalComponents)

"""
=====================
Rule Mining
=====================
"""
data = apriori_x.values.tolist()
a = TransactionEncoder()
a_data = a.fit(data).transform(data)
df = pd.DataFrame(a_data,columns=a.columns_)
change = {False:0, True:1}
df = df.replace(change)
# ===============================
# Applying Apriori and Resulting
# ==============================
df = apriori(df,min_support=0.5, use_colnames=True, verbose=1)
df_ar = association_rules(df,metric='confidence', min_threshold=0.8)
df_ar = df_ar.sort_values(['confidence','lift'], ascending=[False, False])
print(df_ar.to_string())

"""
========================================================
Elbow Method Within cluster Sum of Squared Errors (WSS)
========================================================
"""
sse = {}
for k in range(1, 30):
    kmeans = KMeans(n_clusters=k).fit(PCA_components.iloc[:,:2])
    #X["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.tight_layout()
plt.title('Elbow Method')
plt.show()

"""
========================================================
Silhouette Method for selection of K
========================================================
"""
sil = []
kmax = 20
for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(PCA_components.iloc[:,:2])
    labels = kmeans.labels_
    sil.append(silhouette_score(PCA_components.iloc[:,:2], labels, metric='euclidean'))
plt.figure()
plt.plot(np.arange(2, k + 1, 1), sil, 'bx-')
plt.xticks(np.arange(2,k+1,2))
plt.grid()
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.title('Silhouette Method')
plt.tight_layout()
plt.show()

k = 9
X = copy.copy()
pca = PCA(n_components=2)
X = pca.fit_transform(X)
kmeans = KMeans(n_clusters=k)
y_km = kmeans.fit_predict(X)

for i in range(k):
    plt.scatter(X[y_km == i,0],X[y_km == i,1],label=f'Cluster {i+1}',s=0.7)

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='*')
plt.legend()
plt.grid()
plt.title(f'{k} Kmeans-Clusters')
plt.tight_layout()
plt.show()

end = time.time()
print('================================================================================================')
print(f'Elapsed Time: {end - start}')