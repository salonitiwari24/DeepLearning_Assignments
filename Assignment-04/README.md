# Deep Learning Assignment 4: LSTM-Based Time-Series Forecasting

## Problem Statement

Develop an LSTM-based deep learning model for time-series forecasting using historical stock price data.

## Dataset Overview

- **Dataset:** Apple Inc. (AAPL) Historical Stock Prices
- **Source:** Yahoo Finance
- **Data:** Daily historical stock prices
- **Features:** Date, Open, High, Low, Close, and Volume
- **Target:** Closing price
- **Time Period:** Approximately 5 years of historical data

## Implementation Details

1. **Data Collection & Exploration:**
   - Loaded historical AAPL stock price data.
   - Examined the dataset structure and dimensions.
   - Visualized historical closing prices to observe price trends.

2. **Data Preprocessing:**
   - Selected the daily closing price as the forecasting target.
   - Handled missing values.
   - Applied `MinMaxScaler` to normalize the closing prices between 0 and 1.

3. **Sequence Creation:**
   - Created time-series sequences using the previous 60 trading days.
   - Used these 60-day sequences to predict the closing price of the following day.
   - Split the sequences into 80% training and 20% testing data while preserving chronological order.

4. **LSTM Model:**
   - Developed a Sequential LSTM model using TensorFlow/Keras.
   - Used two LSTM layers with 50 units each.
   - Added Dropout layers to reduce overfitting.
   - Used a Dense output layer for predicting the next closing price.
   - Compiled the model using the Adam optimizer and Mean Squared Error loss.

5. **Training & Evaluation:**
   - Trained the model for 20 epochs with a batch size of 32.
   - Plotted training and validation loss.
   - Generated predictions on the test dataset.
   - Evaluated the model using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
   - Compared actual and predicted stock prices using a visualization.
   - Generated a next-day closing price prediction.

## Technologies Used

- `tensorflow` / `keras` (LSTM, Dense, Dropout)
- `pandas`
- `numpy`
- `scikit-learn` (MinMaxScaler, MAE, RMSE)
- `matplotlib`
- `yfinance`
