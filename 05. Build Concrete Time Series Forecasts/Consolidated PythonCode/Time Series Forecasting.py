#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the basic preprocessing packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# The series module from Pandas will help in creating a time series
from pandas import Series,DataFrame
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# About the Data Set (Location: https://www.kaggle.com/sumanthvrao/daily-climate-time-series-data) 
# To forecast the daily climate of a city in India
time_series = pd.read_csv('DailyDelhiClimateTrain.csv', parse_dates=['date'], index_col='date')
time_series.head()


# In[3]:


# Below are a few statistical methods on time series that will help in understanding the data patterns
# Plotting all the individual columns to observe the pattern of data in each column
time_series.plot(subplots=True)


# In[4]:


# Calculating the mean, maximum values, and minimum of all individual columns of the dataset
time_series.mean()


# In[5]:


time_series.max()


# In[6]:


time_series.min()


# In[7]:


# The describe() method gives information like count, mean, deviations and quartiles of all columns
time_series.describe()


# In[8]:


# Resampling the dataset using the Mean() resample method
timeseries_mm = time_series['wind_speed'].resample("A").mean()
timeseries_mm.plot(style='g--')
plt.show()


# In[9]:


# Calculating the rolling mean with a 14-bracket window between time intervals
time_series['wind_speed'].rolling(window=14, center=False).mean().plot(style='-g')
plt.show()


# In[10]:


#Code Snippet 2: Working with stationarity


# In[11]:


# Importing the basic preprocessing packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# The series module from Pandas will help in creating a time series
from pandas import Series,DataFrame
get_ipython().run_line_magic('matplotlib', 'inline')
# Statsmodel and Adfuller will help in testing the stationarity of the time series
import statsmodels
from statsmodels.tsa.stattools import adfuller

time_series_train = pd.read_csv('DailyDelhiClimateTrain.csv', parse_dates=True)
time_series_train["date"] = pd.to_datetime(time_series_train["date"])
time_series_train.date.freq ="D"
time_series_train.set_index("date", inplace=True)
time_series_train.columns


# In[12]:


# Decomposing the time series with Statsmodels Decompose Method
from statsmodels.tsa.seasonal import seasonal_decompose
sd_1 = seasonal_decompose(time_series_train["meantemp"])
sd_2 = seasonal_decompose(time_series_train["humidity"])
sd_3 = seasonal_decompose(time_series_train["wind_speed"])
sd_4 = seasonal_decompose(time_series_train["meanpressure"])
sd_1.plot()
sd_2.plot()
sd_3.plot()
sd_4.plot()


# In[13]:


# From the above graph’s observations, it looks like everything other than meanpressure is already stationary

# To re-confirm stationarity, we will run all columns through the ad-fuller test
adfuller(time_series_train["meantemp"])


# In[14]:


adfuller(time_series_train["humidity"])


# In[15]:


adfuller(time_series_train["wind_speed"])


# In[16]:


adfuller(time_series_train["meanpressure"])


# In[17]:


# Consolidate the ad-fuller tests to test from static data
temp_var = time_series_train.columns
print('significance level : 0.05')
for var in temp_var:
    ad_full = adfuller(time_series_train[var])
    print(f'For {var}')
    print(f'Test static {ad_full[1]}',end='\n \n')


# In[18]:


# With the ad-fuller test, we can now conclude that all data is stationary since static tests are below significance levels. This also rejects the hypothesis that meanpressure was non-static.

# Let us now move towards training and validating the prediction model
from statsmodels.tsa.vector_ar.var_model import VAR
train_model = VAR(time_series_train)
fit_model = train_model.fit(6)
# AIC is lower for lag_order 6. Hence, we can assume the lag_order of 6.
fix_train_test = time_series_train.dropna()
order_lag_a = fit_model.k_ar
X = fix_train_test[:-order_lag_a]
Y = fix_train_test[-order_lag_a:]

# Model Validation
validate_y = X.values[-order_lag_a:]
forcast_val = fit_model.forecast(validate_y,steps=order_lag_a)
train_forecast = DataFrame(forcast_val,index=time_series_train.index[-order_lag_a:],columns=Y.columns)
train_forecast


# In[19]:


# Check performance of the predictions’ model
from sklearn.metrics import mean_absolute_error
for i in time_series_train.columns:
    print(f'MAE of {i} is {mean_absolute_error(Y[[i]],train_forecast[[i]])}')


# In[20]:


test_forecast = pd.read_csv('DailyDelhiClimateTest.csv',parse_dates=['date'], index_col='date')
period_range = pd.date_range('2017-01-05',periods=6)
order_lag_b = fit_model.k_ar
X1,Y1 = test_forecast[1:-order_lag_b],test_forecast[-order_lag_b:]
input_val = Y1.values[-order_lag_b:]
data_forecast = fit_model.forecast(input_val,steps=order_lag_b)
df_forecast = DataFrame(data_forecast,columns=X1.columns,index=period_range)
df_forecast


# In[21]:


# Plotting the test data with auto correlation
from statsmodels.graphics.tsaplots import plot_acf
# The next 6 periods of mean temperature (graph 1) and wind_speed (graph 2)
plot_acf(df_forecast["meantemp"])
plot_acf(df_forecast["wind_speed"])


# In[22]:


# Code Snippet 3: Granger Causality Tests


# In[23]:


# Import Granger Causality module from the statsmodels package and use the Chi-Squared test metric
from statsmodels.tsa.stattools import grangercausalitytests
test_var = time_series.columns
lag_max = 12
test_type = 'ssr_chi2test'

causal_val = DataFrame(np.zeros((len(test_var),len(test_var))),columns=test_var,index=test_var)
for a in test_var:
    for b in test_var:
        c = grangercausalitytests ( time_series [ [b,a] ], maxlag = lag_max, verbose = False)
        pred_val = [round ( c [ i +1 ] [0] [test_type] [1], 5 ) for i in range (lag_max) ]
        min_value = np.min (pred_val)
        causal_val.loc[b,a] = min_value
causal_val


# In[ ]:




