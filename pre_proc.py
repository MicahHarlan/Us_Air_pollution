import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error



from sklearn.decomposition import PCA

"""
======================
Micah Harlan
This file cleans the read in data 
and adds some features to it 
to make it more useful for ML
======================
"""

read = 'pollution_2000_2022.csv'
df = pd.read_csv(read)
df.drop(columns=['Unnamed: 0'],inplace=True)

"""
======================
Number of Observations
======================
"""
#n = 575000
#test_n = 640000
#df = df.iloc[n:,:]






air_qual = ['Good','Moderate','Unhealthy for Sensitive Groups','Unhealthy','Very Unhealthy']
ranges = [(0,50),(51,100),(101,150),(151,200),(201,300)]

temp = df['O3 AQI']
O3_AQI_class = pd.DataFrame()



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
df.drop_duplicates(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Day'] = df['Date'].dt.day
#df = df.groupby(['Year','Month','State','Day','County','City']).mean(numeric_only=True)
#df.reset_index(inplace=True)
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

"""
==========================================
Highest AQI Value is the actual AQI value.
==========================================
"""
"New Feature Name: Chosen AQI"
"AQI value is chosen from the max AQI value of the 3."
df['AQI'] = df[['O3 AQI','CO AQI','SO2 AQI','NO2 AQI']].max(axis=1)

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
=========
Quad Plot
=========
"""
temp_df = df.groupby(['Year']).mean(numeric_only=True)
temp_df.reset_index(inplace=True)

print(f'{temp_df["Year"]}')
units = {'O3':'PPM','NO2':'PPB','SO2':'PPB','CO':'PPM'}
plt.figure()
plt.tight_layout()
fiq,axs = plt.subplots(2,2)
axs[0,0].plot(temp_df['Year'],temp_df['CO Mean'],'tab:orange')
axs[0,0].set_title('CO Mean PPM 2019-2022')


axs[1,0].plot(temp_df['Year'],temp_df['SO2 Mean'],'tab:red')
axs[1,0].set_title('SO2 Mean PPB 2019-2022')

axs[1,1].plot(temp_df['Year'],temp_df['NO2 Mean'])
axs[1,1].set_title('NO2 Mean PPB 2019-2022')

plt.tight_layout()
plt.show()

print(df.describe())
print(df.isna().sum())

df = df[(df['Date'] >= '2019-01-01')]


"""
========================
Dimensionality Reduction
========================
"""
print(df[['AQI','Y']])

#df.set_index(df['Date'],inplace=True)

X = df.drop(columns=['classification','Address','Date'])

y = X['Y']
X.drop(columns=['Y','County','State'],inplace=True)


le = LabelEncoder()
le.fit(X['NO2_AQI_label'])
X['NO2_AQI_label'] = le.fit_transform(X['NO2_AQI_label'])
le.fit(X['O3_AQI_label'])
X['O3_AQI_label'] = le.fit_transform(X['O3_AQI_label'])
le.fit(X['CO_AQI_label'])
X['CO_AQI_label'] = le.fit_transform(X['CO_AQI_label'])
le.fit(X['SO2_AQI_label'])
X['SO2_AQI_label'] = le.fit_transform(X['SO2_AQI_label'])

numerical = [ 'O3 Mean',
       'O3 1st Max Value', 'O3 1st Max Hour', 'O3 AQI', 'CO Mean',
       'CO 1st Max Value', 'CO 1st Max Hour', 'CO AQI', 'SO2 Mean',
       'SO2 1st Max Value', 'SO2 1st Max Hour', 'SO2 AQI', 'NO2 Mean',
       'NO2 1st Max Value', 'NO2 1st Max Hour', 'NO2 AQI','AQI']


std = StandardScaler()
for s in numerical:
    std.fit(X[s].to_numpy().reshape(len(X[s]),-1))
    X[s] = std.fit_transform(X[s].to_numpy().reshape(len(X[s]),-1))

std.fit(y.to_numpy().reshape(len(y),-1))
y  = std.fit_transform(y.to_numpy().reshape(len(y),-1))


#X.drop(columns=['Sate'],inplace=True)
X = pd.get_dummies(X,drop_first=True)
print(X)
copy_of_x = X
"""
======================
Down Sampling
======================
"""

"""sns.countplot(x=df['AQI'],data=df)
plt.title('Countplot of Target')
plt.tight_layout()
plt.show()
"""


#X.drop(columns=['NO2_AQI_label','SO2_AQI_label','CO_AQI_label','Season','Year'
#    ,'CO 1st Max Value','CO AQI','CO 1st Max Value'],inplace=True)


#X = X[['O3 AQI','AQI','O3 Mean','NO2 Mean','O3_AQI_label','SO2 Mean',]]

rf = RandomForestRegressor()
X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=False,test_size=.2 )

rf.fit(X_train,y_train)

importances = rf.feature_importances_

features = X.columns
indices = np.argsort(importances)


"""
=====================================
Feature importance threshold dropping
=====================================
"""

threshold = 0.05
dropped = []

kept = []
importance =  []
for ind in indices:
    if importances[ind] < threshold:
        dropped.append(features[ind])
    else:
        kept.append(features[ind])
        importance.append(importances[ind])

X.drop(columns=dropped,inplace=True)

print(X)
plt.title('Feature Importance')
plt.barh(range(len(indices)),importances[indices],color='b',align='center')
plt.yticks(range(len(indices)),[features[i] for i in indices])
plt.xlabel('Relative Importance.')
plt.tight_layout()
plt.legend()
plt.show()

rf_pred = rf.predict(X_test)
print(f'MSE: {mean_squared_error(y_test,rf_pred)}')

plt.plot(np.arange(0,len(rf_pred)),rf_pred,label='Predicted')
plt.plot(np.arange(0,len(rf_pred)),y_test,label='Actual')
plt.xticks()
plt.title('Actual, versus predicted values')
plt.legend()
plt.grid()
plt.show()

"""
=======
Values dropped from Random Forest Analysis
=======
"""

columns=['NO2_AQI_label','SO2_AQI_label','CO_AQI_label','Season','Year'
    ,'CO 1st Max Value','CO AQI','CO 1st Max Value']

"""
PRINCIPAL COMPONENT ANALYSIS
"""
X = copy_of_x
pca = PCA(n_components='mle',svd_solver='full')
pca.fit(X)
X_pca = pca.transform(X)
PCA(n_components='mle',svd_solver='full')

print(f'Explained Variance Ratio {pca.explained_variance_ratio_}')
s = 0
a = 0
for i in pca.explained_variance_ratio_:
    s += i
    a += 1
    if s >= .90:
        break
print(f'Number of Features explaining 90% dependent variance variance: {a}')
n_features = len(X.columns)

plt.figure()
plt.plot(np.arange(1,len(np.cumsum(pca.explained_variance_ratio_))
+1,1),np.cumsum(pca.explained_variance_ratio_),label='Variance Ratio')
#plt.axvline(x=7.8,color='r')
#plt.axhline(0.9,color='g')

plt.xticks(np.arange(1,len(np.cumsum(pca.explained_variance_ratio_))+1,1))
plt.xlabel('N Components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA Exact 90% Threshold v-line and h-line')
plt.legend()
plt.grid()
plt.show()
