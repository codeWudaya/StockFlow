import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

# Define the start and end dates
start = '2010-01-01'
end = '2019-12-31'

st.title("Stock Trend Prediction")

# Predefined list of stock tickers
stock_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']

# Specify the stock symbol using a dropdown
user_input = st.selectbox("Select stock Ticker", stock_tickers)

# Fetch the data using yfinance
df = yf.download(user_input, start, end)

# describe data
st.subheader(f'Data for {user_input} from 2010 - 2019 ')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price v/s Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

# Visualizations 100 days MA
st.subheader('Closing Price v/s Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

# Visualizations 200 days MA
st.subheader('Closing Price v/s Time Chart with 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r', label='100 MA')
plt.plot(ma200, 'b', label='200 MA')
plt.plot(df.Close, 'g', label='Closing Price')
plt.legend()
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


#load my model
from keras.models import load_model

model = load_model('keras_model.h5')

#testing part

past_100 = data_training.tail(100)
final_df = pd.concat([past_100, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test,y_test=np.array(x_test),np.array(y_test)

y_pred = model.predict(x_test)

sl=scaler.scale_
scale_facter = 1/sl[0]

y_pred=y_pred*scale_facter

y_test=y_test*scale_facter


# Final Graph
st.subheader('Prediction v/s Original')

fig_final = plt.figure(figsize=(12, 6))

plt.plot(y_test, 'b', label="Original Price")
plt.plot(y_pred, 'r', label="Predicted Price")

plt.xlabel("Time")
plt.ylabel("Price")

plt.legend()
st.pyplot(fig_final)