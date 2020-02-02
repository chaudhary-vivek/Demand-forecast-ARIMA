#Importing libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm

#Importing the data
df=pd.read_csv('perrin-freres-monthly-champagne-.csv')

#The data contains NaN in the last two rows
df.drop([105,106], axis=0, inplace=True)

#Naming the columns in the dataset month and sales
df.columns = ['Month', 'Sales']

#Converting the first column into date-time format
df['Month']=pd.to_datetime(df['Month'])

#Adding index column to the data
df.set_index('Month',inplace=True)

#Plotting the graph
df.plot()

#The graph shows seasonality annualy 

#Making ARIMA model
model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 0, 0),seasonal_order=(1,1,1,12))
results=model.fit()

#Forecasting from 90th row to 103rd row
df['forecast']=results.predict(start=90,end=103,dynamic=True)

#Plotting the forecast and the actual data from 90th to 103rd row
df[['Sales','forecast']].plot(figsize=(12,8))

#The prediction appears consistent with the actual value


#Creating a data offset by 24 months
from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]

#Creating a dataframe from the future dates
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)

#Concatenating the above dataframe with the earlier dataframe
future_df=pd.concat([df,future_datest_df])

#Predicting values from 104th to 128th row
future_df['forecast']=results.predict(start=104,end=128,dynamic=True)

#Plotting the predicted values of the future with the exsisting values
future_df[['Sales', 'forecast']].plot(figsize=(12,8))

#The predicted values seem consistent with the trend

