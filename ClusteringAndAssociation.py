import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


"DELETE"
#from sklearnex import patch_sklearn
#patch_sklearn()
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler,LabelEncoder,Normalizer,normalize

import warnings
warnings.filterwarnings("ignore")
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

"PPM = Parts Per Million"
"PPB = Parts Per Billion"
units = {'O3':'PPM','NO2':'PPB','S02':'PPB','CO2':'PPM'}


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


#%%
"""
Setting Date
"""
shuffle = False
#df = df[(df['Date'] >= '2015-01-01')]
df['days_since_start'] = (df['Date'] - pd.to_datetime('2000-01-01')).dt.days
df.reset_index(inplace=True,drop=True)

print(len(df))


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
print(X.keys())

print(apriori_x)


copy = X.copy()
X2 = X.copy()
X3 = X.copy()
print(len(X.keys()))

pca = PCA()
principalComponents = pca.fit_transform(X)
PCA_components = pd.DataFrame(principalComponents)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.show()


"""
=====================
Rule Mining
=====================
"""
data = list(df["products"].apply(lambda x:x.split(",") ))
print(data)
a = TransactionEncoder()
a_data = a.fit(data).transform(data)
df = pd.DataFrame(a_data,columns=a.columns_)
print(df)
change = {False:0, True:1}
df = df.replace(change)
print(df)
# ===============================
# Applying Apriori and Resulting
# ==============================
df = apriori(df,min_support=0.2, use_colnames=True, verbose=1)
print(df)
df_ar = association_rules(df,metric='confidence', min_threshold=0.6)
df_ar = df_ar.sort_values(['confidence','lift'], ascending=[False, False])
print(df_ar.head())



"""
========================================================
Elbow Method Within cluster Sum of Squared Errors (WSS)
========================================================
"""

'''
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

k = 6
X = copy.copy()
pca = PCA(n_components=2)
X = pca.fit_transform(X)
kmeans = KMeans(n_clusters=k)
y_km = kmeans.fit_predict(X)

for i in range(k):
    plt.scatter(X[y_km == i,0],X[y_km == i,1],label=f'Cluster {i+1}',s=10)

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='*')
plt.legend()
plt.grid()
plt.title(f'{k} Kmeans-Clusters')
plt.tight_layout()
plt.show()
'''