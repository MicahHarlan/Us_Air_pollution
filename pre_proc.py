import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

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

#df = df.groupby(['State','Date','County']).mean(numeric_only=True)
#df.reset_index(inplace=True)

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
df['N02_AQI_label'] = np.select([df['NO2 AQI'].between(0,50),
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




"""temp_df = df.groupby(['Year','Season'
                      ]).mean(numeric_only=True)
temp_df.reset_index(inplace=True)
plt.figure()
plt.tight_layout()
plt.plot(temp_df['Year'],temp_df['CO Mean'])
plt.show()"""


"""
Ratio of Pollutants: 
Calculate ratios between different pollutant levels,
like O3/NO2 or CO/SO2,
to explore potential interactions between pollutants.
"""







"""
========================
Dimensionality Reduction
========================
"""



X = df.drop(columns=['classification','Address','Date'])
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

le.fit(X['N02_AQI_label'])
X['N02_AQI_label'] = le.fit_transform(X['N02_AQI_label'])

le.fit(X['O3_AQI_label'])
X['O3_AQI_label'] = le.fit_transform(X['O3_AQI_label'])

le.fit(X['CO_AQI_label'])
X['CO_AQI_label'] = le.fit_transform(X['CO_AQI_label'])

le.fit(X['SO2_AQI_label'])
X['SO2_AQI_label'] = le.fit_transform(X['SO2_AQI_label'])


"""
======================
Down Sampling
======================
"""
sns.countplot(x=df['AQI'],data=df)
plt.title('Countplot of Target')
plt.tight_layout()
plt.show()





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
