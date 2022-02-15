from cProfile import label
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2019-12-31'

stock = st.text_input('Enter your stock code', 'AAPL')

df = pdr.DataReader(stock, 'yahoo', start, end)

st.subheader('Stock code: ' + stock)

st.subheader('Data from 2010 - 2019')
st.write(df.describe())

st.subheader('Closing Price chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price')
st.pyplot(fig)

st.subheader('Closing Price with MA100 and MA200 chart')
fig = plt.figure(figsize=(12, 6))
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
plt.plot(ma100, color='r', label='MA100')
plt.plot(ma200, color='g', label='MA200')
plt.plot(df['Close'], color='b', label='Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
st.pyplot(fig)

num_data = int(df.shape[0] * 0.7)

#Take test data to predict
df_train = pd.DataFrame(df['Close'][0: num_data])
df_test = pd.DataFrame(df['Close'][num_data:])

#Scale price from 0 to 1
df_test = df_train.tail(100).append(df_test)
scaler = MinMaxScaler(feature_range=(0, 1))
df_test = scaler.fit_transform(df_test)

X_test, y_test = [], []

for i in range(100, df_test.shape[0]):
  X_test.append(df_test[i-100:i])
  y_test.append(df_test[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

#load model
model = load_model('stock_prediction.h5')
predict = model.predict(X_test)

#re-scale again to original price
scale_factor = 1 / scaler.scale_[0]
predict = predict * scale_factor
y_test = y_test * scale_factor

st.subheader('Actual Price vs Predicted Price chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(y_test, color='g', label='Actual Price')
plt.plot(predict, color='b', label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
st.pyplot(fig)
