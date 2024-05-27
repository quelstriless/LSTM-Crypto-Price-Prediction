# LSTM-Crypto-Price-Prediction

Using Long Short-Term Memory (LSTM) to predict future prices of RUNE/USDT using historical data and technical indicators like SMA, EMA, MACD, and RSI.

## Overview

This project utilizes a machine learning model (LSTM) to forecast the future prices of the RUNE/USDT trading pair. The model is trained using historical price data and technical indicators to improve its predictive accuracy.

## Data Source

**Binance API for RUNE/USDT data**

To learn more about the Binance API and how to create API keys, visit: [Binance API Documentation](https://www.binance.com/en/support/faq/how-to-create-api-keys-on-binance-360002502072)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/quelstriless/LSTM-Crypto-Price-Prediction.git
    cd LSTM-Crypto-Price-Prediction
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    
## Usage

1. Ensure you have your Binance API keys.
2. Edit the script to include your API keys.
3. Run the script:
    ```sh
    python script_name.py
    ```

## Results

![Future Price Prediction](https://github.com/quelstriless/LSTM-Crypto-Price-Prediction/assets/71846076/b47d9e03-7fab-43b5-b3ca-5ea01a91a0b3)

The above plot shows the predicted future prices of RUNE/USDT.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
