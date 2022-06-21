#!/usr/bin/env python
# coding: utf-8

# In[43]:


pip install yfinance


# In[44]:


pip install pandas-datareader 


# In[45]:


pip install keras


# In[46]:


pip install tensorflow


# In[47]:


pip install plotly


# In[48]:



import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import yfinance  as yf
import numpy as np
import seaborn as sns 
import math
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, classification_report
import datetime as dt
import time
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

 


# In[49]:


data = pd.read_csv('C:/Users/HP/Downloads/BTC.csv', header=0, index_col='Date', parse_dates=True)
data


# In[50]:


data.head()


# In[51]:


data.tail()


# In[52]:


data.describe()


# In[53]:


data


# In[54]:


# Visualize the closing price history 
plt.figure(figsize=(16,8))
plt.title('close price history')
plt.plot(data['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close price Bitcoin',fontsize=18)
plt.show()


# In[55]:


# create a new dataframe with only close column
data=data.filter(['Close'])

# convert dataframe to a numpy array
dataset=data.values

# get the number of rows to train the model on
training_data_len= math.ceil(len(dataset)*.8)


# In[56]:


training_data_len


# In[57]:


# Scale the data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)


# In[58]:


scaled_data


# In[59]:


# Create the trainig dataset
# Create the scaled trainig dataset
train_data=scaled_data[0:training_data_len,:]
# Split the data into x_train and y_train datasets
x_train=[]
y_train=[]


for i in range(60 ,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=61:
        print(x_train)
        print(y_train)
        print()


# In[60]:


# Convert x_train and y_train to a numpy array
x_train,y_train=np.array(x_train),np.array(y_train)
x_train.shape


# In[61]:


#  Reshape the data
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape


# In[62]:


# Build the lSTM model
model= Sequential()
model.add(LSTM(50, return_sequences=True ,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False ))
model.add(Dense(25))
model.add(Dense(1))


# In[63]:


# Compile the model
model.compile(optimizer='adam' ,loss='mean_squared_error')


# In[64]:


model.fit(x_train,y_train,batch_size=10, epochs=50)


# In[75]:


model.save('keras_model.h5')


# In[76]:


# Create the testing dataset
test_data=scaled_data[training_data_len-60:,:]
# Create the dataset x_test and y_test
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60 ,len(test_data)):
    x_test.append(test_data[i-60:i,0])


# In[77]:


# Convert the data to a numpy array
x_test=np.array(x_test)


# In[78]:


# Reshape the data
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
x_test.shape


# In[79]:


# Get the model predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[80]:


predictions.shape


# In[81]:


# Get the root mean squared error
rmse = np.sqrt((np.mean(predictions -y_test)**2))
rmse


# In[94]:


train= data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions']=predictions

fig = px.line(train['Close'], title='Model historique')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
                     dict(count=1, label="1m", step="month", stepmode="backward"),
                     dict(count=6, label="6m", step="month", stepmode="backward"),
                     dict(count=1, label="1y", step="year", stepmode="todate"),
                     dict(step="all")
        ])
    )
)

fig.show()
plt.savefig('modhist.svg')


# In[97]:



fig = px.line(valid['Predictions'], title='Prediction')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
                     dict(count=1, label="1m", step="month", stepmode="backward"),
                     dict(count=6, label="5m", step="month", stepmode="backward"),
                     dict(count=1, label="1y", step="year", stepmode="todate"),
                     dict(step="all")
        ])
    )
)

fig.show()
plt.savefig('modpred.svg')


# In[96]:


# Show the valid and predictions prices
valid


# In[ ]:





# In[ ]:





# In[ ]:




