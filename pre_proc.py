
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler,LabelEncoder,Normalizer,normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
n_jobs=-1

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

"""
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


#%%
"""
Setting Date
"""
shuffle = False
df = df[(df['Date'] >= '2016-01-01')]
df['days_since_start'] = (df['Date'] - pd.to_datetime('2016-01-01')).dt.days
df.reset_index(inplace=True,drop=True)
#df.to_csv('classification.csv',index=False)
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
copy_of_x = X.copy()
print(X.keys())

"""
=========
Dropping low variance features
==========
"""
X.drop(low_variance, inplace=True,axis=1)
copy_of_x = X.copy()

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

"""predictions = rf.predict(X_test)
predictions_rev = std.inverse_transform(predictions.reshape(len(X_test),1))
actual = std.inverse_transform(y_test.to_numpy().reshape(len(X_test),1))
point1 = actual.min()
point2 = actual.max()
plt.figure(figsize=(15,5))
sns.lineplot(x=np.arange(0,len(predictions),1),y=predictions_rev.reshape(len(X_test),),label='Predicted')
sns.lineplot(x=np.arange(0,len(actual),1),y=actual.reshape(len(X_test),),label='Actual',alpha=0.4,color='red')
plt.xlabel('N Observations')
plt.ylabel('Actual Value')
plt.title(f'Random Forest: Actual vs. Predicted Value RMSE: {round(mean_squared_error(actual,predictions_rev,squared=False),2)}')
plt.legend()
plt.show()"""

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
print(f'Number of Features explaining 90% dependent variance variance: {a}')
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
#X.to_csv('cleaned_aqi.csv',index=False)