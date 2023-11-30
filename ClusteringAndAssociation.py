import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
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
X = df.drop(columns=['Address','Date','State','County','City'],axis=1)

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

X2 = X.copy()

"""
========================================================
Elbow Method Within cluster Sum of Squared Errors (WSS)
========================================================
"""
sse = {}
for k in range(1, 30):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X)
    X["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
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
kmax = 100

for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X2)
    labels = kmeans.labels_
    sil.append(silhouette_score(X2, labels, metric='euclidean'))
plt.figure(figsize=(16,16))
plt.plot(np.arange(2, k + 1, 1), sil, 'bx-')
plt.xticks(np.arange(2,k+1,1))
plt.grid()
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.title('Silhouette Method')
plt.show()
