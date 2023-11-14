import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler,LabelEncoder,Normalizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor


"""
Need to add The data from pre_poc.py
"""

"""
===============================
backward stepwise
regression needs to move to phase 2
===============================
"""
X = copy_of_x
X.dropna(inplace=True)
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

print(f'SVD FEATURES REMOVED: {len(removed)}')
svd_removed = removed
print(model.summary())

predictions = model.predict(X_test)
predictions_rev = std.inverse_transform(predictions.to_numpy().reshape(len(X_test),1))
actual = std.inverse_transform(y_test.to_numpy().reshape(len(X_test),1))

sns.lineplot(x=np.arange(0,len(predictions),1),y=predictions_rev.reshape(len(X_test),),label='Predicted')
sns.lineplot(x=np.arange(0,len(actual),1),y=actual.reshape(len(X_test),),label='Actual',alpha=0.4,color='red')
plt.xlabel('N Observations')
plt.ylabel('Actual Value')
plt.title(f'SVD: Actual vs. Predicted Value MSE: {round(mean_squared_error(actual,predictions_rev),2)}')
plt.legend()
plt.show()

#Linear Regression?
#Deccision Tree Regression

