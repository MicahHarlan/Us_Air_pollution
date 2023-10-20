import pandas as pd
import numpy as np
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

#df = df.groupby(['State','Date','County']).mean()
#df.reset_index(inplace=True)

df['Season'] = np.select([df['Month'].between(3,5),
                        df['Month'].between(6,8),
                        df['Month'].between(9,11),
                        df['Month'] == 12,
                        df['Month'].between(1,2)],
                         ['Spring','Summer','Fall','Winter','Winter']
                         )

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



"Forecast Increase Rate of Change. Per Year"


"""
Ratio of Pollutants: 
Calculate ratios between different pollutant levels,
like O3/NO2 or CO/SO2,
to explore potential interactions between pollutants.
"""


'Year, Season, Month'
'Federal Holiday T or F' 'MAYBE'