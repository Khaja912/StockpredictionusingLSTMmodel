#!/usr/bin/env python
# coding: utf-8

# ## AAPL Stock

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as dt  # it is for online site data cache like tiingo,ticker
import yfinance as yf
#import warnings
#warnings.filterwarnings("ignore") 


# In[2]:


start='2010-01-01'
end = '2023-01-24'
df= yf.download('AAPL',start, end)   
df.head()


# In[3]:


df.tail()


# In[4]:


df=df.reset_index()
df.head()


# In[5]:


df=df.drop(['Date','Adj Close'],axis=1)
df.head()


# In[6]:


# we are working for Closing Stock
plt.plot(df.Close)
plt.show()


# In[7]:


# Creating Moving Average
df.head()


# In[8]:


# finding 100 moving average 
ma100=df.Close.rolling(100).mean()
ma100


# In[9]:


plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'red')
plt.show()


# In[10]:


# finding 200 moving average 
ma200=df.Close.rolling(200).mean()
ma200


# In[11]:


plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'red')
plt.plot(ma200,'g')
plt.show()


# In[12]:


df.shape


# In[13]:


# splitting data into training and testing

df_train= pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
df_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(df_train.shape)
print(df_test.shape)


# In[14]:


df_train.head()


# In[15]:


df_test.head()


# In[16]:


# Scale down the data
from sklearn.preprocessing import MinMaxScaler       # import lib
scaler = MinMaxScaler(feature_range=(0,1))         # call function


# In[17]:


df_train_arr= scaler.fit_transform(df_train)
df_train_arr


# In[18]:


x_train = []
y_train = []

for i in range(100,df_train_arr.shape[0]):        # df_train_arr[101:1]   # here we take only from df_train
    x_train.append(df_train_arr[i-100: i])
    y_train.append(df_train_arr[i,0])
    
x_train


# In[19]:


x_train,y_train = np.array(x_train),np.array(y_train)
print(np.array(x_train))


# In[20]:


x_train.shape[1]


# ### ML Model

# In[21]:


from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


# In[22]:


model = Sequential()
model.add(LSTM(units =50,activation='relu',return_sequences=True,
               input_shape=(x_train.shape[1],1)))                      # working on 1 column last one(close column) 
model.add(Dropout(0.2))

#2nd layer
model.add(LSTM(units =60,activation='relu',return_sequences=True))                     
model.add(Dropout(0.3))

# third layer
model.add(LSTM(units =80,activation='relu',return_sequences=True))                     
model.add(Dropout(0.4))

# last layer
model.add(LSTM(units =120,activation='relu'))                     
model.add(Dropout(0.5))

# DENSE LAYER
model.add(Dense(units=1))


# In[23]:


model.summary()


# In[24]:


model.compile(optimizer='adam',loss='mean_squared_error')    # loss for classification is categorical/binary cross entropy
model.fit(x_train,y_train,epochs=50)


# In[25]:


model.save('keras_model.h5')


# In[47]:


train_predict=model.predict(x_train)
test_predict=model.predict(x_test)


# In[48]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[49]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[50]:


math.sqrt(mean_squared_error(y_test,test_predict))


# ## Test

# In[51]:


df_test


# In[52]:


# but we want the the previous 100 days of data from the testing data means in df_test past_100_days
# so we get from the 


# In[53]:


# these 100 days values add to testing dataset for finding moving average    and in the
past_100_days = df_train.tail(100)
past_100_days


# In[54]:


final_df_test = past_100_days.append(df_test,ignore_index=True)
final_df_test

# now this is final df for testing ( with 100 days training data + testing dataset )


# In[55]:


print(df_test.shape)
print(df_train.shape)
print(final_df_test.shape)      # 100 days add in this


# ## Scaling down 

# In[56]:


input_data= scaler.fit_transform(final_df_test)
print(input_data)
input_data.shape       # 951+100


# In[57]:


x_test =[]
y_test= []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
print(input_data.shape[0])


# In[58]:


x_test,y_test = np.array(x_test),np.array(y_test)
print(x_test.shape)
print(y_test.shape)


# In[59]:


# Making Prediction 
y_predicted = model.predict(x_test)


# In[68]:


# making x_train predict and comapring both
x_train_pred=model.predict(x_train)


# In[70]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,x_train_pred))


# In[71]:


math.sqrt(mean_squared_error(y_test,y_predicted))


# In[60]:


y_predicted.shape


# In[61]:


y_test


# In[62]:


y_predicted


# In[63]:


print(y_test.shape)
print(x_test.shape)    


# ## Scaling up

# In[64]:


scaler.scale_


# In[65]:


scale_factor= 1/scaler.scale_
y_predicted = y_predicted * scale_factor
y_test= y_test* scale_factor


# In[66]:


y_predicted


# In[67]:


y_test


# In[ ]:





# In[ ]:





# In[43]:


plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original Price')
plt.plot(y_predicted,'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# ## predicting next 30 days

# In[ ]:





# In[ ]:





# In[ ]:




