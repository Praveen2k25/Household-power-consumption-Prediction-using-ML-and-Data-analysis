import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import keras

st.write(f"TensorFlow version: {tf.__version__}")
st.write(f"Keras version: {keras.__version__}")

st.title('Household Electricity Consumption Forecasting')

# Load data
df = pd.read_csv("C:\\Users\\ASUS\\Desktop\\GitHub\\House power consumption\\household_power_consumption.csv")
st.write("Dataset columns")
st.write(df.columns)

if 'Date' in df.columns and 'Time' in df.columns:
    df['DateTime'] = df['Date'] + ' ' + df['Time']

    def parse_datetime(x):
        for fmt in ('%d/%m/%Y %H:%M:%S', '%m/%d/%Y %I:%M:%S %p', '%d-%m-%Y %H:%M:%S', '%m-%d-%Y %I:%M:%S %p'):
            try:
                return pd.to_datetime(x, format=fmt)
            except ValueError:
                continue
        return pd.NaT

    df['DateTime'] = df['DateTime'].apply(parse_datetime)

    st.write("DateTime parsing completed")
    st.write(df['DateTime'].head())

    df.dropna(subset=['DateTime'], inplace=True)
    df.set_index('DateTime', inplace=True)

    # Check for the existence of columns before dropping
    columns_to_drop = ['Date', 'Time', 'Unnamed: 9', 'Unnamed: 10']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=existing_columns_to_drop, inplace=True)

    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)

    columns_to_convert = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    df[columns_to_convert] = df[columns_to_convert].astype(float)

    st.write("Dataframe after preprocessing")
    st.write(df.head())

    st.subheader('Exploratory Data Analysis (EDA)')
    st.write("Global Active Power Over Time")
    st.line_chart(df['Global_active_power'])

    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour

    st.write("Monthly Global Active Power Consumption")
    fig, ax = plt.subplots()
    sns.lineplot(x='month', y='Global_active_power', data=df, ax=ax)
    st.pyplot(fig)

    st.write("Hourly Global Active Power Consumption")
    fig, ax = plt.subplots()
    sns.lineplot(x='hour', y='Global_active_power', data=df, ax=ax)
    st.pyplot(fig)

    st.write("Completed EDA")

    train_data, test_data = train_test_split(df['Global_active_power'], test_size=0.2, shuffle=False)

    sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_fit = sarima_model.fit(disp=False)
    sarima_pred = sarima_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=False)

    train_data_lstm = train_data.values.reshape(-1, 1)
    test_data_lstm = test_data.values.reshape(-1, 1)

    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data_lstm)
    test_data_scaled = scaler.transform(test_data_lstm)

    X_train, y_train = [], []

    for i in range(60, len(train_data_scaled)):
        X_train.append(train_data_scaled[i-60:i, 0])
        y_train.append(train_data_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Using Functional API
    inputs = Input(shape=(X_train.shape[1], 1))
    x = LSTM(units=50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(units=50, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(units=1)(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=1, batch_size=32)  # Increased batch size to speed up training

    # Making predictions on the test data
    X_test = []

    for i in range(60, len(test_data_scaled)):
        X_test.append(test_data_scaled[i-60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    st.write("Model training and prediction completed")

    df['Lag_1'] = df['Global_active_power'].shift(1)
    df['Lag_7'] = df['Global_active_power'].shift(7)
    df.dropna(inplace=True)

    mae = mean_absolute_error(test_data, sarima_pred)
    rmse = np.sqrt(mean_squared_error(test_data, sarima_pred))
    st.write(f'SARIMA Model MAE: {mae}')
    st.write(f'SARIMA Model RMSE: {rmse}')

    mae_lstm = mean_absolute_error(test_data[60:], predictions)
    rmse_lstm = np.sqrt(mean_squared_error(test_data[60:], predictions))
    st.write(f'LSTM Model MAE: {mae_lstm}')
    st.write(f'LSTM Model RMSE: {rmse_lstm}')

    st.write("Model evaluation completed")

    future_steps = 30
    sarima_forecast = sarima_fit.get_forecast(steps=future_steps)
    future_dates = [df.index[-1] + pd.Timedelta(days=x) for x in range(0, future_steps)]
    sarima_forecast_values = sarima_forecast.predicted_mean

    st.subheader('Future Electricity Consumption Forecast')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Global_active_power'], label='Historical Data')
    ax.plot(future_dates, sarima_forecast_values, label='SARIMA Forecast')
    ax.set_title('Future Electricity Consumption Forecast')
    ax.set_xlabel('DateTime')
    ax.set_ylabel('Global Active Power (kilowatts)')
    ax.legend()
    st.pyplot(fig)

    df.to_csv('preprocessed_household_power_consumption.csv')
    model.save('lstm_model.h5')
    sarima_fit.save('sarima_model.pkl')

    st.write("Script execution complete.")
else:
    st.write("The dataset does not contain the expected 'Date' and 'Time' columns.")
