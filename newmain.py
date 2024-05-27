import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

# Initialize Binance API
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
client = Client(api_key, api_secret)

# Fetch historical data
def fetch_data(symbol, interval, start_str):
    klines = client.get_historical_klines(symbol, interval, start_str)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return data

# Add technical indicators
def add_indicators(data):
    data['SMA'] = data['close'].rolling(window=15).mean()
    data['EMA'] = data['close'].ewm(span=15, adjust=False).mean()
    data['MACD'] = data['close'].ewm(span=12, adjust=False).mean() - data['close'].ewm(span=26, adjust=False).mean()
    data['RSI'] = compute_rsi(data['close'], 14)
    data.dropna(inplace=True)
    return data

def compute_rsi(series, period):
    delta = series.diff().dropna()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

# Preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['close', 'SMA', 'EMA', 'MACD', 'RSI']])
    
    # Separate scaler for closing prices
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    data['close_scaled'] = close_scaler.fit_transform(data[['close']])
    
    return scaled_data, scaler, close_scaler

# Create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(seq_length, len(data)):
        sequences.append(data[i-seq_length:i])
        labels.append(data[i][0])  # We want to predict the 'close' price
    sequences = np.array(sequences)
    sequences = np.reshape(sequences, (sequences.shape[0], sequences.shape[1], data.shape[1]))
    labels = np.array(labels)
    return sequences, labels

# Build LSTM model with regularization
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict future prices
def predict_future_prices(model, last_sequence, steps, scaler):
    future_predictions = []
    current_sequence = last_sequence.reshape((1, -1, last_sequence.shape[1]))
    
    for _ in range(steps):
        next_step = model.predict(current_sequence)
        future_predictions.append(next_step.flatten()[0])
        next_sequence = np.roll(current_sequence, -1, axis=1)
        next_sequence[:, -1, 0] = next_step
        current_sequence = next_sequence
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

# Main workflow
symbol = 'RUNEUSDT'
interval = '1h'
start_str = '1 Jan, 2021'
data = fetch_data(symbol, interval, start_str)
data = add_indicators(data)
scaled_data, scaler, close_scaler = preprocess_data(data)
seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build and train the model
model = build_model((seq_length, X.shape[2]))

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train, y_train, batch_size=10, epochs=3, validation_split=0.1, verbose=1, callbacks=[early_stopping])

# Predict future prices using the newly trained model
future_steps = 200  # Predicting the next 24 hours
last_sequence = scaled_data[-seq_length:]
future_prices = predict_future_prices(model, last_sequence, future_steps, close_scaler)

# Prepare data for plotting
train = data.iloc[:train_size + seq_length]
valid = data.iloc[train_size + seq_length:].copy()
valid['Predictions'] = close_scaler.inverse_transform(model.predict(X_test))[:, 0]

future_dates = pd.date_range(start=valid.index[-1], periods=future_steps + 1, inclusive='right')
future_df = pd.DataFrame(index=future_dates, data=future_prices, columns=['Predictions'])

# Plotting
plt.figure(figsize=(16, 8))
plt.plot(data.index, data['close'], label='Historical Close Price')
plt.plot(valid.index, valid['Predictions'], label='Validation Predictions')
plt.plot(future_df.index, future_df['Predictions'], label='Future Predictions', linestyle='--')
plt.title('Price Prediction Model with Enhanced Features and Regularization')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
