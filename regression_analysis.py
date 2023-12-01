import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from prettytable import PrettyTable
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

'''
Plotting Predicted Values
'''
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
ols_r2 = model.rsquared
ols_adj_r2 = model.rsquared_adj
ols_AIC = model.aic
ols_BIC = model.bic
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

rf_r2 = r2_score(actual, predictions_rev)
rf_adj_r2 = 1 - (1 - rf_r2) * ((len(X_test) - 1) / (len(X_test) - len(X_test.keys()) - 1))
rf_AIC = 2222222
rf_BIC = 222
rf_MSE = round(mean_squared_error(actual,predictions_rev),2)

table = PrettyTable()

# Define table columns
table.field_names = ['Model','R-squared', 'adjusted R-square', 'AIC', 'BIC','MSE']

# Add data to the table
table.add_row(["Random Forest", rf_r2, rf_adj_r2,rf_AIC,rf_BIC,rf_MSE])
table.add_row(tab)
print(table)

'''
plotting predicted values
'''
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


'''
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
'''