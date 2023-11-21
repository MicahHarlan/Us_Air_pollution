import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler,LabelEncoder,Normalizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings("ignore")


df = pandas.read_csv('cleaned_aqi.csv')
y = df['Y']
X = df.drop('Y',axis=1)

shuffle = False

y_std = StandardScaler()
y_std.fit(y.to_numpy().reshape(len(y),-1))
y = y_std.fit_transform(y.to_numpy().reshape(len(y),-1))

std= StandardScaler()
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
copy_x = X.copy()

"""
===============================
backward stepwise
regression needs to move to phase 2
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

predictions = model.predict(X_test)
predictions_rev = y_std.inverse_transform(predictions.to_numpy().reshape(len(X_test),1))
actual = y_std.inverse_transform(y_test.reshape(len(X_test),1))

sns.lineplot(x=np.arange(0,len(predictions),1),y=predictions_rev.reshape(len(X_test),),label='Predicted')
sns.lineplot(x=np.arange(0,len(actual),1),y=actual.reshape(len(X_test),),label='Actual',alpha=0.4,color='red')
plt.xlabel('N Observations')
plt.ylabel('Actual Value')
plt.title(f'Backwards Stepwise: Actual vs. Predicted Value MSE: {round(mean_squared_error(actual,predictions_rev),2)}')
plt.legend()
plt.show()

"""
Run a Random Forest Again and put selected features in OLS.summary
then compare each one in a pretty table HW#3
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

sns.lineplot(x=np.arange(0,len(predictions),1),y=predictions_rev.reshape(len(X_test),),label='Predicted')
sns.lineplot(x=np.arange(0,len(actual),1),y=actual.reshape(len(X_test),),label='Actual',alpha=0.4,color='red')
plt.xlabel('N Observations')
plt.ylabel('Actual Value')
plt.title(f'Random Forest: Actual vs. Predicted Value MSE: {round(mean_squared_error(actual,predictions_rev),2)}')
plt.legend()
plt.show()


"""
Prediction interval using stepwise Regression With selected 
Features
"""
