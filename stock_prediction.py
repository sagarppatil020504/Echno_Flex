import requests
from textblob import TextBlob
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import schedule
import time
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# --- Configurations ---
HISTORY_WINDOW = 60  # Number of days for the LSTM input sequence
FEATURES = ['Close', 'Volume']

# Get the current time
current_time = datetime.now()
previous_day = current_time - timedelta(days=1)
# --- Functions ---

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    try:
        stock = yf.download(ticker, start=start_date, end=end_date)
        return stock
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

def add_technical_indicators(df):
    """Add technical indicators to the DataFrame."""
    df['SMA'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    return df

def calculate_rsi(series, window=14):
    """Calculate the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def perform_sentiment_analysis(news_articles):
    """Perform sentiment analysis on a list of news articles."""
    sentiments = []
    for article in news_articles:
        analysis = TextBlob(article)
        sentiments.append(analysis.sentiment.polarity)
    return np.mean(sentiments)

def prepare_data_for_lstm(data, history_window):
    """Prepare data for LSTM input."""
    x, y = [], []
    for i in range(history_window, len(data)):
        x.append(data[i-history_window:i])
        y.append(data[i])
    return np.array(x), np.array(y)

def build_lstm_model(input_shape):
    """Build and compile the LSTM model."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def fetch_live_stock_data(ticker):
    """Fetch live stock data for predictions."""
    try:
        stock = yf.download(ticker, period='1d', interval='1m')
        return stock
    except Exception as e:
        print(f"Error fetching live stock data: {e}")
        return None

def visualize_predictions(actual, predicted):
    """Plot actual vs predicted prices."""
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual Price', color='blue')
    plt.plot(predicted, label='Predicted Price', color='red')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Function to predict future stock prices
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

# Example: Fetch sentiment for Apple (AAPL)
# live_sentiment = fetch_live_news_sentiment("AAPL")

#some features
def get_macro_factors():
    # Replace this with actual macroeconomic data sources
    return {"interest_rate": 6.1, "inflation": 2.2, "gdp_growth": 2.7}

def prepare_live_sequence(live_data, seq_length=30):
    """
    Prepare live data for prediction.
    """
    x_live = []
    for i in range(seq_length, len(live_data)):
        x_live.append(live_data[i-seq_length:i])
    return np.array(x_live)

# Live Prediction Pipeline with Future Prediction
def live_prediction_pipeline_with_future(ticker,future_steps=30):
    """
    Automate live stock price prediction, including future price prediction.
    """
    # Fetch live data
    live_data = fetch_live_stock_data("AAPL", interval="1m", period="1d")
    sentiment = fetch_live_news_sentiment(ticker)
    model = build_lstm_model()
        
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
    scaler = MinMaxScaler()
    
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

    # Visualization
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

    # Calculate RMSE for live predictions
    rmse = np.sqrt(np.mean((live_predictions - actual_prices) ** 2))
    print(f"Live Prediction RMSE: {rmse}")

    # Debugging: Print future predictions
    print(f"Future Predictions for the next {future_steps} intervals:\n", future_predictions)

# # Schedule the pipeline to run every minute
# schedule.every(1).minutes.do(lambda: live_prediction_pipeline_with_future(future_steps=30))

# while True:
#     schedule.run_pending()
#     time.sleep(1)

def main(ticker):
    # ticker = "AAPL"
    start_date = "2015-01-01"
    end_date = str(previous_day.strftime('%Y-%m-%d'))
    data = fetch_stock_data(ticker, start_date, end_date)

    if data is not None:
        # Add technical indicators
        data = add_technical_indicators(data)
        data.dropna(inplace=True)

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[FEATURES])

        # Prepare data for LSTM
        x, y = prepare_data_for_lstm(scaled_data[:, 0], HISTORY_WINDOW)
        x = x.reshape((x.shape[0], x.shape[1], 1))

        # Build and train the model
        model = build_lstm_model(x.shape[1:])
        model.fit(x, y, epochs=20, batch_size=32, validation_split=0.2)

        # Predict on historical data
        predictions = model.predict(x)
        predictions = scaler.inverse_transform(predictions)

        # Visualize the results
        visualize_predictions(data['Close'][HISTORY_WINDOW:], predictions)

        # Fetch and predict live data
        live_data = fetch_live_stock_data(ticker)
        if live_data is not None:
            scaled_live_data = scaler.transform(live_data[FEATURES])
            live_x = scaled_live_data[-HISTORY_WINDOW:].reshape(1, HISTORY_WINDOW, 1)
            live_prediction = model.predict(live_x)
            live_prediction = scaler.inverse_transform(live_prediction)
            print(f"Live Predicted Price: {live_prediction[0][0]}")
    else:
        print("Failed to fetch historical stock data.")

# --- Main Workflow ---
if __name__ == "__main__":
    # Fetch historical data
    ##
    main("AAPL")