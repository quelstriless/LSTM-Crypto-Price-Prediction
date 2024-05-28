import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization, Embedding
from keras.layers import MultiHeadAttention, Add, GlobalAveragePooling1D
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import mplfinance as mpf

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

# Create sequences for Transformer
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(seq_length, len(data)):
        sequences.append(data[i-seq_length:i])
        labels.append(data[i][0])  # We want to predict the 'close' price
    sequences = np.array(sequences)
    labels = np.array(labels)
    return sequences, labels

# Transformer Model
def transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Positional Encoding
    positions = np.arange(input_shape[0])
    positions = Embedding(input_shape[0], input_shape[1])(positions)
    
    # Adding Positional Encoding to inputs
    x = inputs + positions
    
    # Multi-Head Attention
    x = LayerNormalization(epsilon=1e-6)(x)
    attention_output = MultiHeadAttention(num_heads=8, key_dim=input_shape[1])(x, x)
    x = Add()([x, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Feed Forward
    feed_forward = Dense(256, activation='relu')(x)
    feed_forward = Dense(input_shape[1])(feed_forward)
    x = Add()([x, feed_forward])
    
    # Global Average Pooling
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    
    # Output Layer
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Predict future prices
def predict_future_prices(model, last_sequence, steps, scaler):
    future_predictions = []
    current_sequence = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))

    for _ in range(steps):
        next_step = model.predict(current_sequence)
        future_predictions.append(next_step.flatten()[0])
        
        # Update the sequence with the new prediction
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
model = transformer_model((seq_length, X.shape[2]))

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.1, verbose=1, callbacks=[early_stopping])

# Predict future prices using the newly trained model
future_steps = 200  # Predicting the next 200 hours
last_sequence = scaled_data[-seq_length:]
future_prices = predict_future_prices(model, last_sequence, future_steps, close_scaler)

# Prepare data for plotting
train = data.iloc[:train_size + seq_length]
valid = data.iloc[train_size + seq_length:].copy()
valid['Predictions'] = close_scaler.inverse_transform(model.predict(X_test))[:, 0]

future_dates = pd.date_range(start=valid.index[-1], periods=future_steps + 1, inclusive='right')
future_df = pd.DataFrame(index=future_dates, data=future_prices, columns=['Predictions'])

# Combine historical data with future predictions
all_data = pd.concat([data, future_df])

# Add the prediction lines
plt.plot(valid.index, valid['Predictions'], label='Validation Predictions', color='orange')
plt.plot(future_df.index, future_df['Predictions'], label='Future Predictions', linestyle='--', color='purple')
plt.legend()
plt.show()
