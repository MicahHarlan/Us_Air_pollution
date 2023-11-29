import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder,PolynomialFeatures
from sklearn.model_selection import train_test_split,TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from prettytable import PrettyTable
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR
import warnings
warnings.filterwarnings("ignore")
n_jobs = -1

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
print(X)
tscv = TimeSeriesSplit(n_splits=5)
copy_x = X.copy()


"""
===============================
backward stepwise
===============================
"""
X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=shuffle,test_size=.2 )
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

plt.figure(figsize=(10,4))
fig,axs = plt.subplots(2,1)
predictions = model.predict(X_test)
predictions_rev = y_std.inverse_transform(predictions.to_numpy().reshape(len(X_test),1))
actual = y_std.inverse_transform(y_test.reshape(len(X_test),1))

sns.lineplot(ax=axs[0],x=np.arange(0,len(predictions),1),y=predictions_rev.reshape(len(X_test),),label='Predicted',color='blue')
sns.lineplot(ax=axs[0],x=np.arange(0,len(actual),1),y=actual.reshape(len(X_test),),label='Actual',alpha=0.4,color='green')
axs[0].set_xlabel('N Observations')
axs[0].set_ylabel('Actual Value')
axs[0].set_title(f'Backwards Stepwise: Actual vs. Predicted Value RMSE: {round(mean_squared_error(actual,predictions_rev,squared=False),2)}')

test_predictions = model.predict(X_train)
test = y_std.inverse_transform(test_predictions.to_numpy().reshape(len(test_predictions),1))
actual2 = y_std.inverse_transform(y_train.reshape(len(y_train),1))
sns.lineplot(ax=axs[1],x=np.arange(0,len(test),1),y=test.reshape(len(test),),label='Train Set Predicted',color='blue')
sns.lineplot(ax=axs[1],x=np.arange(0,len(actual2),1),y=actual2.reshape(len(y_train),),label='Train Actual',alpha=0.4,color='green')
axs[1].set_xlabel('N Observations')
axs[1].set_ylabel('Actual Value')
axs[1].set_title(f'Backwards Stepwise: Actual vs. Predicted Value RMSE: {round(mean_squared_error(actual2,test,squared=False),2)}')
fig.tight_layout()
fig.show()

"""
=========================================================
EXTRACTING R-squared, adjusted R-square, AIC, BIC and MSE + RMSE
=========================================================
"""
ols_r2 = model.rsquared
ols_adj_r2 = model.rsquared_adj
ols_AIC = model.aic
ols_BIC = model.bic
ols_MSE = round(mean_squared_error(actual,predictions_rev),2)

"""
====================================
RANDOM FOREST FOR FEATURE SELECTION
====================================
"""
X = copy_x.copy()
rf = RandomForestRegressor(max_depth=10)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2 ,shuffle=shuffle)
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
print(model.summary())

rf.fit(X_train,y_train)
predictions = rf.predict(X_test)
predictions_rev = y_std.inverse_transform(predictions.reshape(len(X_test),1))
actual = y_std.inverse_transform(y_test.reshape(len(X_test),1))

plt.figure(figsize=(10,4))
sns.lineplot(x=np.arange(0,len(predictions),1),y=predictions_rev.reshape(len(X_test),),label='Predicted',color='blue')
sns.lineplot(x=np.arange(0,len(actual),1),y=actual.reshape(len(X_test),),label='Actual',alpha=0.4,color='green')
plt.xlabel('N Observations')
plt.ylabel('Actual Value')
plt.title(f'Random Forest: Actual vs. Predicted Value RMSE: {round(mean_squared_error(actual,predictions_rev,squared=False),2)}')
plt.legend()
plt.show()

"""
==========================
Dropping Stepwise Features
==========================
"""
print(f'Stepwise Removed {removed}')
copy_x = copy_x.copy().drop(removed,axis=1)
X = copy_x.copy()

"""
===========================
Prediction interval using 
stepwise Regression With selected 
Features
===========================
"""

"""
============================
Random Forest GridSearch
============================
"""

"""
param_grid = {'criterion':['squared_error', 'absolute_error','absolute_error'],
              'max_features':['sqrt', 'log2']}

RF_GRID = GridSearchCV(RandomForestRegressor(max_depth=10,n_jobs=n_jobs),param_grid,verbose=0,
                     n_jobs=n_jobs,scoring='neg_mean_squared_error',cv=tscv)

RF_GRID.fit(X,y.ravel())
grid_1 =RF_GRID.best_params_
print(RF_GRID.best_params_)
print(RF_GRID.best_score_)
"""


#{'criterion': 'squared_error', 'max_features': 'sqrt'}
param_grid = {'ccp_alpha':np.arange(0,5,1),'min_impurity_decrease':np.arange(0,5,1)}
RF_GRID = GridSearchCV(RandomForestRegressor(max_depth=10,n_jobs=n_jobs,max_features='sqrt',
                                             criterion='squared_error')
                       ,param_grid,verbose=0,
                     n_jobs=n_jobs,scoring='neg_mean_squared_error',cv=tscv)
RF_GRID.fit(X,y.ravel())

print(RF_GRID.best_params_)
print(RF_GRID.best_score_)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,shuffle=shuffle)
RF_GRID = RandomForestRegressor(max_depth=10,n_jobs=n_jobs,max_features='sqrt',
                                criterion='squared_error',ccp_alpha=RF_GRID.best_params_['ccp_alpha'],
                                min_impurity_decrease=RF_GRID.best_params_['min_impurity_decrease'])

RF_GRID.fit(X_train,y_train)
predictions = RF_GRID.predict(X_test)
predictions_rev = y_std.inverse_transform(predictions.reshape(len(predictions),1))
actual = y_std.inverse_transform(y_test.reshape(len(y_test),1))
print(f'RF with best Params MSE: {round(mean_squared_error(actual,predictions_rev,squared=False),2)}')

plt.figure(figsize=(10,4))
sns.lineplot(x=np.arange(0,len(predictions),1),y=predictions_rev.reshape(len(X_test),),label='Predicted',color='blue')
sns.lineplot(x=np.arange(0,len(actual),1),y=actual.reshape(len(X_test),),label='Actual',alpha=0.4,color='green')
plt.xlabel('N Observations')
plt.ylabel('Actual Value')
plt.title(f'Best Params Random Forest: Actual vs. Predicted Value RMSE: {round(mean_squared_error(actual,predictions_rev,squared=False),2)}')
plt.legend()
plt.show()

"""
============================
Polynomial Regression Here
============================
"""
"""
X.drop('O3_AQI_label',inplace=True,axis=1)
param_grid = {'polynomialfeatures__degree':np.arange(1,3,1)}

def PolyNomialRegression(degree=2):
    return make_pipeline(PolynomialFeatures(degree),LinearRegression(n_jobs=n_jobs))

poly_grid = GridSearchCV(PolyNomialRegression(),param_grid
                         ,scoring='neg_mean_squared_error',
                         verbose=0,n_jobs=n_jobs,cv=tscv)
poly_grid.fit(X,y)
print(poly_grid.best_params_)
pr = PolynomialFeatures(degree=poly_grid.best_params_['polynomialfeatures__degree'])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2 ,shuffle=shuffle)
X_train = pr.fit_transform(X_train)
X_test = pr.fit_transform(X_test)

model = sm.OLS(y_train,X_train).fit()
predictions = model.predict(X_test)
predictions_rev = y_std.inverse_transform(predictions.reshape(len(predictions),1))
actual = y_test
actual = y_std.inverse_transform(actual.reshape(len(actual),1))
print(round(mean_squared_error(actual,predictions_rev,squared=False),2))

plt.figure(figsize=(10,4))
sns.lineplot(x=np.arange(0,len(predictions),1),y=predictions_rev.reshape(len(X_test),),label='Predicted',color='blue')
sns.lineplot(x=np.arange(0,len(actual),1),y=actual.reshape(len(X_test),),label='Actual',alpha=0.4,color='green')
plt.xlabel('N Observations')
plt.ylabel('Actual Value')
plt.title(f'Polynomial: Actual vs. Predicted Value RMSE: {round(mean_squared_error(actual,predictions_rev,squared=False),2)}')
plt.legend()
plt.show()"""

"""
{'polynomialfeatures__degree': 1}
"""



"""
=========
Quad Plot
=========
"""
"""temp_df = df.groupby(['Year']).mean(numeric_only=True)
temp_df.reset_index(inplace=True)

#print(f'{temp_df["Year"]}')
units = {'O3':'PPM','NO2':'PPB','SO2':'PPB','CO':'PPM'}
plt.figure()
plt.tight_layout()
fiq,axs = plt.subplots(2,2)
axs[0,0].plot(temp_df['Year'],temp_df['CO Mean'],'tab:orange')
axs[0,0].set_title('CO Mean PPM')

axs[0,1].plot(temp_df['Year'],temp_df['O3 Mean'],'tab:green')
axs[0,1].set_title('O3 Mean PPM')

axs[1,0].plot(temp_df['Year'],temp_df['SO2 Mean'],'tab:red')
axs[1,0].set_title('SO2 Mean PPB')

axs[1,1].plot(temp_df['Year'],temp_df['NO2 Mean'])
axs[1,1].set_title('NO2 Mean PPB')
plt.tight_layout()
plt.show()
"""