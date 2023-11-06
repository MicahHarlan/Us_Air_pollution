import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import mean_squared_error




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

axs[0,1].plot(temp_df['Year'],temp_df['O3 Mean'],'tab:green')
axs[0,1].set_title('O3 Mean PPM 2019-2022')

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
print(df)

rus = RandomUnderSampler()
#X,y = rus.fit_resample(X,df['classification'])

y = X['Y']
X.drop(columns=['Y'],inplace=True)

le = LabelEncoder()

le.fit(X['State'])
X['State'] = le.fit_transform(X['State'])

le.fit(X['County'])
X['County'] = le.fit_transform(X['County'])

le.fit(X['City'])
X['City'] = le.fit_transform(X['City'])

le.fit(X['Season'])
X['Season'] = le.fit_transform(X['Season'])

le.fit(X['NO2_AQI_label'])
X['NO2_AQI_label'] = le.fit_transform(X['NO2_AQI_label'])

le.fit(X['O3_AQI_label'])
X['O3_AQI_label'] = le.fit_transform(X['O3_AQI_label'])

le.fit(X['CO_AQI_label'])
X['CO_AQI_label'] = le.fit_transform(X['CO_AQI_label'])

le.fit(X['SO2_AQI_label'])
X['SO2_AQI_label'] = le.fit_transform(X['SO2_AQI_label'])


#X = pd.get_dummies(data=X,drop_first=True)

std = StandardScaler()
numerical = ['AQI','O3 Mean'
    ,'O3 AQI','NO2 1st Max Hour','SO2 Mean','NO2 Mean','O3 AQI']


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


X.drop(columns=['NO2_AQI_label','SO2_AQI_label','CO_AQI_label','Season','Year'
    ,'CO 1st Max Value','CO AQI','CO 1st Max Value'],inplace=True)


#X = X[['O3 AQI','AQI','O3 Mean','NO2 Mean','O3_AQI_label','SO2 Mean',]]

rf = RandomForestRegressor()
X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=False,test_size=.2 )

rf.fit(X_train,y_train)

importances = rf.feature_importances_

features = X.columns
indices = np.argsort(importances)
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