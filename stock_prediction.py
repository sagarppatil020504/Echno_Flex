import requests
from textblob import TextBlob
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import schedule
import time
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

def fetch_live_stock_data(ticker, interval="1m", period="1d"):
    """
    Fetch live stock data for a given ticker.
    """
    live_data = yf.download(tickers=ticker, interval=interval, period=period)
    live_data['Date'] = live_data.index
    live_data.reset_index(drop=True, inplace=True)
    return live_data

#fetching old data and preprocessing it

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Date'] = data.index
    data.reset_index(drop=True, inplace=True)
    return data

def fetch_live_news_sentiment(ticker):
    """
    Fetch live news and perform sentiment analysis.
    # """
    API_KEY = '6685d1357e634d04819fd7c35134ec03'
    url = f'https://newsapi.org/v2/everything?q={ticker}&to={current_time}0&sortBy=publishedAt&apiKey={API_KEY}'
    # url=f"https://serpapi.com/search.json?engine=google_news&q={ticker}&gl=in&hl=en"
    response = requests.get(url).json()
    articles = response.get('articles', [])
    print(response)
    sentiments = []
    for article in articles:
        text = f"{article['title']} {article.get('description', '')}"
        sentiment = TextBlob(text).sentiment.polarity
        sentiments.append(sentiment)

    # Return average sentiment if articles are found
    return np.mean(sentiments) if sentiments else 0

# Fetch old news data for sentiment analysis for prediction of old dataset
def get_news_sentiment(keyword, start_date, end_date):
    # Example API: Replace with your News API Key
    # Getting data from marketaux free api key
    API_KEY = '6685d1357e634d04819fd7c35134ec03'
    url = f'https://newsapi.org/v2/everything?q={ticker}&to={current_time}0&sortBy=publishedAt&apiKey={API_KEY}'
    response = requests.get(url).json()
    articles = response.get('articles', [])

    sentiments = []
    for article in articles:
        title = article['title']
        description = article.get('description', '')
        text = f"{title} {description}"
        sentiment = TextBlob(text).sentiment.polarity
        sentiments.append(sentiment)
    return np.mean(sentiments) if sentiments else 0

def get_macro_factors():
    # Replace this with actual macroeconomic data sources
    return {"interest_rate": 6.1, "inflation": 2.2, "gdp_growth": 2.7}

# Create time-series data for LSTM
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i, 0])  # Predict the 'Close' price
    return np.array(x), np.array(y)

# Prepare live data sequence for prediction
def prepare_live_sequence(live_data, seq_length=30):
    """
    Prepare live data for prediction.
    """
    x_live = []
    for i in range(seq_length, len(live_data)):
        x_live.append(live_data[i-seq_length:i])
    return np.array(x_live)

def live_prediction_pipeline(ticker):
    """
    Automate live stock price prediction.
    """
    # Fetch live data
    live_data = fetch_live_stock_data(ticker, interval="1m", period="1d")
    sentiment = fetch_live_news_sentiment(ticker
                                          )

    # Add technical indicators to live_data
    live_data['SMA_20'] = live_data['Close'].rolling(window=20).mean()
    live_data['RSI'] = 100 - (100 / (1 + live_data['Close'].pct_change().rolling(window=14).mean() /
                                          live_data['Close'].pct_change().rolling(window=14).std()))
    live_data['MACD'] = live_data['Close'].ewm(span=12).mean() - live_data['Close'].ewm(span=26).mean()
    print("Live Data:\n", live_data.tail())
    # Add macroeconomic factors (ensure macro_factors is preloaded)
    macro_factors = {
        'interest_rate': 5.5,  # Example value
        'inflation': 6.0,      # Example value
        'gdp_growth': 4.0      # Example value
    }
    live_data['Sentiment'] = sentiment
    live_data['Interest_Rate'] = macro_factors['interest_rate']
    live_data['Inflation'] = macro_factors['inflation']
    live_data['GDP_Growth'] = macro_factors['gdp_growth']

    # Handle missing values
    live_data.fillna(method='bfill', inplace=True)

    # Debugging: Print live data
    print("Live Data:\n", live_data.tail())

    # Preprocess the data for the model
    features = ['Close', 'SMA_20', 'RSI', 'MACD', 'Sentiment', 'Interest_Rate', 'Inflation', 'GDP_Growth']
    scaled_data = scaler.fit_transform(live_data[features])
    x_live = prepare_live_sequence(scaled_data)

    # Make predictions
    live_predictions = model.predict(x_live)
    live_predictions = scaler.inverse_transform(
    np.concatenate((live_predictions, np.zeros((live_predictions.shape[0], 7))), axis=1))[:, 0]

    # Get the actual stock prices for comparison
    actual_prices = live_data['Close'].values[-len(live_predictions):]
    
    rmse = np.sqrt(np.mean((live_predictions - actual_prices) ** 2))
    print(f"Live Prediction RMSE: {rmse}")


    # predictions = model.predict(x_live)
    # predictions = scaler.inverse_transform(
    #     np.concatenate((predictions, np.zeros((predictions.shape[0], len(features) - 1))), axis=1))[:, 0]

    # # Compare with actual prices
    # actual_prices = live_data['Close'].values[-len(predictions):]
    
    
    return live_predictions ,actual_prices
    

def predict_future_prices(model, last_sequence, future_steps=30):
    """
    Predict future stock prices based on the last sequence of scaled data.
    :param model: Trained LSTM model
    :param last_sequence: Most recent scaled data sequence
    :param future_steps: Number of future steps to predict
    :return: Predicted future prices
    """
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(future_steps):
        # Predict the next price
        next_price = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))

        # Append the prediction to the list
        future_predictions.append(next_price[0, 0])
        # Update the current sequence
        next_price = np.zeros((1, current_sequence.shape[1]))
        print(next_price)
        next_sequence = np.append(current_sequence[1:], next_price, axis=0)
        current_sequence = next_sequence
        print(current_sequence)
        print(next_sequence)

    return np.array(future_predictions)

def live_prediction_pipeline_with_future(ticker,future_steps,param):
    """
    Automate live stock price prediction, including future price prediction.
    """
    # Fetch live data
    live_data = fetch_live_stock_data(ticker, interval="1m", period="1d")

    # Check if live_data is None
    if live_data is None:
        print("Error fetching live data. Exiting.")
        return None  # Return None to signal an error

    sentiment = fetch_live_news_sentiment(ticker)

    # Check if sentiment is None
    if sentiment is None:
        print("Error fetching sentiment. Exiting.")
        return None  # Return None to signal an error

    # # Fetch live data
    # live_data = fetch_live_stock_data("AAPL", interval="1m", period="1d")
    # sentiment = fetch_live_news_sentiment("AAPL")

    # Add technical indicators to live_data
    live_data['SMA_20'] = live_data['Close'].rolling(window=20).mean()
    live_data['RSI'] = 100 - (100 / (1 + live_data['Close'].pct_change().rolling(window=14).mean() /
                                          live_data['Close'].pct_change().rolling(window=14).std()))
    live_data['MACD'] = live_data['Close'].ewm(span=12).mean() - live_data['Close'].ewm(span=26).mean()

    # Add macroeconomic factors
    macro_factors = get_macro_factors()
    live_data['Sentiment'] = sentiment
    live_data['Interest_Rate'] = macro_factors['interest_rate']
    live_data['Inflation'] = macro_factors['inflation']
    live_data['GDP_Growth'] = macro_factors['gdp_growth']

    # Handle missing values
    live_data.fillna(method='bfill', inplace=True)

    # Preprocess the data for the model
    features = ['Close', 'SMA_20', 'RSI', 'MACD', 'Sentiment', 'Interest_Rate', 'Inflation', 'GDP_Growth']
    scaled_data = scaler.fit_transform(live_data[features])
    x_live = prepare_live_sequence(scaled_data)

    # Predict current stock prices
    live_predictions = model.predict(x_live)
    live_predictions = scaler.inverse_transform(
        np.concatenate((live_predictions, np.zeros((live_predictions.shape[0], len(features) - 1))), axis=1))[:, 0]

    # Get the actual stock prices for comparison
    actual_prices = live_data['Close'].values[-len(live_predictions):]

    # Predict future prices
    last_sequence = x_live[-1]  # Get the last sequence of live data
    future_predictions_scaled = predict_future_prices(model, last_sequence, future_steps)
    future_predictions = scaler.inverse_transform(
        np.concatenate((future_predictions_scaled.reshape(-1, 1),
                        np.zeros((future_steps, len(features) - 1))), axis=1))[:, 0]

    # Calculate RMSE for live predictions
    rmse = np.sqrt(np.mean((live_predictions - actual_prices) ** 2))
    print(f"Live Prediction RMSE: {rmse}")

    # Debugging: Print future predictions
    print(f"Future Predictions for the next {future_steps} intervals:\n", future_predictions)
    
    # Visualization
    if param ==True:
        plt.figure(figsize=(14, 7))
        plt.plot(actual_prices, label="Actual Prices", color="blue")
        plt.plot(live_predictions, label="Predicted Prices (Live)", color="red")
        plt.plot(range(len(actual_prices), len(actual_prices) + len(future_predictions)),
                future_predictions, label="Predicted Prices (Future)", color="green", linestyle="--")
        plt.title(f"Live and Future Stock Price Prediction for {ticker}")
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.show()
    
    return actual_prices,live_predictions, future_predictions



ticker = "AAPL"
current_time = datetime.now() # Get the current time
previous_day = current_time - timedelta(days=1)    # Subtract 1 day
# Convert to timestamp
# previous_day_timestamp = previous_day.timestamp()

# Fetch stock data
start_date = "2015-01-01"
end_date = str(previous_day.strftime('%Y-%m-%d'))
stock_data = get_stock_data(ticker, start_date, end_date)

#preprocessing the dyata
# Add technical indicators
stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
stock_data['RSI'] = 100 - (100 / (1 + stock_data['Close'].pct_change().rolling(window=14).mean() /
                                      stock_data['Close'].pct_change().rolling(window=14).std()))
stock_data['MACD'] = stock_data['Close'].ewm(span=12).mean() - stock_data['Close'].ewm(span=26).mean()


stock_data['Sentiment'] = stock_data['Date'].apply(
    lambda x: get_news_sentiment(ticker, x.strftime('%Y-%m-%d'), x.strftime('%Y-%m-%d'))
)                                                                                              # Add sentiment analysis
stock_data['Sentiment'] = pd.to_numeric(stock_data['Sentiment'], errors='coerce')

# Add macroeconomic factors
macro_factors = get_macro_factors()
stock_data['Interest_Rate'] = macro_factors['interest_rate']
stock_data['Inflation'] = macro_factors['inflation']
stock_data['GDP_Growth'] = macro_factors['gdp_growth']

# Drop NaN values
# Instead of dropping all NaN values, drop only rows where 'Close' is NaN
# This ensures you have data for scaling
stock_data.dropna( inplace=True)


# Fill remaining NaNs with 0 if any exist to avoid issues in scaling
stock_data.fillna(0, inplace=True)

#sectoin 6


# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(stock_data[['Close', 'SMA_20', 'RSI', 'MACD', 'Sentiment',
                                               'Interest_Rate', 'Inflation', 'GDP_Growth']])


seq_length = 30
x, y = create_sequences(scaled_data, seq_length)

# Split into training and testing datasets
train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]         #taking the old data to predict the live 
y_train, y_test = y[:train_size], y[train_size:]


"""
# live_stock_data = fetch_live_stock_data(ticker, interval="1m", period="1d")
# # Example: Fetch sentiment for Apple (AAPL)
# live_sentiment = fetch_live_news_sentiment(ticker)
# # Preprocess live stock data
# live_stock_data['SMA_20'] = live_stock_data['Close'].rolling(window=20).mean()
# live_stock_data['RSI'] = 100 - (100 / (1 + live_stock_data['Close'].pct_change().rolling(window=14).mean() /
#                                             live_stock_data['Close'].pct_change().rolling(window=14).std()))
# live_stock_data['MACD'] = live_stock_data['Close'].ewm(span=12).mean() - live_stock_data['Close'].ewm(span=26).mean()
# live_stock_data['Sentiment'] = live_sentiment

# # Fill missing values
# live_stock_data.fillna(method='bfill', inplace=True)

# # Add macroeconomic factors
# macro_factors = get_macro_factors()
# live_stock_data['Interest_Rate'] = macro_factors['interest_rate']
# live_stock_data['Inflation'] = macro_factors['inflation']
# live_stock_data['GDP_Growth'] = macro_factors['gdp_growth']

# # Scale live data
# scaler = MinMaxScaler()
# scaled_live_data = scaler.fit_transform(live_stock_data[['Close', 'SMA_20', 'RSI', 'MACD', 'Sentiment','Interest_Rate', 'Inflation', 'GDP_Growth']])


# seq_length = 60
# x_live = prepare_live_sequence(scaled_live_data)
"""

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

live_predictions ,actual_prices = live_prediction_pipeline(ticker)
# Make predictions

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label="Actual Prices", color="blue")
plt.plot(live_predictions, label="Predicted Prices", color="red")
plt.title(f"Live Stock Price Prediction for {ticker}")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

live_prediction_pipeline_with_future(ticker,future_steps=30,param=True)

# # Schedule the pipeline to run every minute
# schedule.every(1).minutes.do(live_prediction_pipeline)


# # Schedule the pipeline to run every minute
# schedule.every(1).minutes.do(lambda: )

# while True:
#     schedule.run_pending()
#     time.sleep(1)
