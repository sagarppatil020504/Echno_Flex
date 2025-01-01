from newsapi import NewsApiClient
from transformers import pipeline

# Initialize News API (replace 'your_api_key' with your News API key)
newsapi = NewsApiClient(api_key='6685d1357e634d04819fd7c35134ec03')

def get_gold_related_news():
    # Fetch gold-related news
    articles = newsapi.get_everything(q='gold price', language='en', sort_by='relevancy', page_size=50)
    headlines = [article['title'] for article in articles['articles']]
    return headlines

# Fetch news headlines
headlines = get_gold_related_news()
print("Headlines:", headlines)


# Load the BERT sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment_with_bert(headlines):
    sentiment_scores = []
    for headline in headlines:
        result = sentiment_analyzer(headline)
        # Assign +1 for positive, -1 for negative
        sentiment = 1 if result[0]['label'] == 'POSITIVE' else -1
        sentiment_scores.append(sentiment * result[0]['score'])  # Multiply by confidence score
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

# Analyze sentiment
sentiment_score = analyze_sentiment_with_bert(headlines)
print(f"Sentiment Score: {sentiment_score}")

# Adjust predictions based on sentiment
adjustment_factor = 0.01  # Adjust prices by 1% for strong sentiment
adjusted_prices = []

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import yfinance as yf
import matplotlib.pyplot as plt
import time
# -------------------------------------
# Step 1: Data Collection
# -------------------------------------

# Fetch historical gold price data
def get_gold_price_data(start_date, end_date):
    gold_data = yf.download('GC=F', start=start_date, end=end_date)  # 'GC=F' is the Yahoo Finance ticker for gold futures
    gold_data['Date'] = gold_data.index
    gold_data.reset_index(drop=True, inplace=True)
    return gold_data

# Fetch data
start_date = "2010-01-01"
end_date = time.strftime("%Y-%m-%d",time.localtime())
gold_data = get_gold_price_data(start_date, end_date)

# -------------------------------------
# Step 2: Feature Engineering
# -------------------------------------

# Create technical indicators
gold_data['SMA_20'] = gold_data['Close'].rolling(window=20).mean()  # Simple Moving Average (20 days)
gold_data['SMA_50'] = gold_data['Close'].rolling(window=50).mean()  # Simple Moving Average (50 days)
gold_data['RSI'] = 100 - (100 / (1 + gold_data['Close'].pct_change().rolling(window=14).mean() /
                                      gold_data['Close'].pct_change().rolling(window=14).std()))
gold_data['MACD'] = gold_data['Close'].ewm(span=12).mean() - gold_data['Close'].ewm(span=26).mean()

# Drop NaN values (caused by rolling calculations)
gold_data.dropna(inplace=True)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(gold_data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD']])

# Create time-series data for the LSTM model
def create_sequences(data, seq_length=60):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i, 0])  # Predict the 'Close' price
    return np.array(x), np.array(y)

seq_length = 60
x, y = create_sequences(scaled_data, seq_length)

# Split into training and testing datasets
train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# -------------------------------------
# Step 3: Build the LSTM Model
# -------------------------------------

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)  # Predict a single value (gold price)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))

# -------------------------------------
# Step 4: Evaluate the Model
# -------------------------------------

# Predict on the test dataset
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(np.concatenate((predicted_prices, np.zeros((predicted_prices.shape[0], 4))), axis=1))[:, 0]


#added set to modify the predicted values
for price in predicted_prices:
    if sentiment_score > 0.2:  # Positive sentiment
        adjusted_price = price * (1 + adjustment_factor * sentiment_score)
    elif sentiment_score < -0.2:  # Negative sentiment
        adjusted_price = price * (1 + adjustment_factor * sentiment_score)
    else:  # Neutral sentiment
        adjusted_price = price
    adjusted_prices.append(adjusted_price)

# Actual prices
actual_prices = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 4))), axis=1))[:, 0]

# Calculate RMSE
rmse = np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))
print(f"RMSE: {rmse}")

# -------------------------------------
# Step 5: Visualization
# -------------------------------------

plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label="Actual Gold Prices", color='blue')
plt.plot(predicted_prices, label="Predicted Gold Prices", color='red')
plt.title("Gold Price Prediction")
plt.xlabel("Time")
plt.ylabel("Gold Price")
plt.legend()
plt.show()

# -------------------------------------
# Step 6: Save the Model
# -------------------------------------

# Save the model in HDF5 format
model.save("lstm_gold_price_model.h5")
print("Model saved successfully!")
































# from newsapi import NewsApiClient
# from transformers import pipeline
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout
# import yfinance as yf
# import matplotlib.pyplot as plt
# import time
# from datetime import timedelta, datetime

# # Initialize News API (replace with your News API key)
# newsapi = NewsApiClient(api_key='6685d1357e634d04819fd7c35134ec03')

# # Function to fetch gold-related news headlines
# def get_gold_related_news():
#     articles = newsapi.get_everything(q='gold price', language='en', sort_by='relevancy', page_size=50)
#     return [article['title'] for article in articles['articles']]

# # Load sentiment analysis model
# sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# # Analyze sentiment of headlines
# def analyze_sentiment(headlines):
#     sentiment_scores = []
#     for headline in headlines:
#         result = sentiment_analyzer(headline)
#         sentiment = 1 if result[0]['label'] == 'POSITIVE' else -1
#         sentiment_scores.append(sentiment * result[0]['score'])
#     return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

# # Fetch historical gold price data
# def get_gold_price_data(start_date, end_date):
#     gold_data = yf.download('GC=F', start=start_date, end=end_date)  # Yahoo Finance ticker for gold futures
#     gold_data['Date'] = gold_data.index
#     gold_data.reset_index(drop=True, inplace=True)
#     return gold_data

# # Create sequences for LSTM
# def create_sequences(data, seq_length=60):
#     x, y = [], []
#     for i in range(seq_length, len(data)):
#         x.append(data[i - seq_length:i])
#         y.append(data[i, 0])  # Predict the 'Close' price
#     return np.array(x), np.array(y)

# # Main process
# def main():
#     # Step 1: Fetch news headlines and analyze sentiment
#     headlines = get_gold_related_news()
#     sentiment_score = analyze_sentiment(headlines)
#     adjustment_factor = 0.01  # Adjust prices by 1% for sentiment

#     # Step 2: Fetch historical gold price data
#     start_date = "2010-01-01"
#     end_date = time.strftime("%Y-%m-%d", time.localtime())
#     gold_data = get_gold_price_data(start_date, end_date)

#     # Feature engineering
#     gold_data['SMA_20'] = gold_data['Close'].rolling(window=20).mean()
#     gold_data['SMA_50'] = gold_data['Close'].rolling(window=50).mean()
#     gold_data['RSI'] = 100 - (100 / (1 + gold_data['Close'].pct_change().rolling(window=14).mean() /
#                                       gold_data['Close'].pct_change().rolling(window=14).std()))
#     gold_data['MACD'] = gold_data['Close'].ewm(span=12).mean() - gold_data['Close'].ewm(span=26).mean()
#     gold_data.dropna(inplace=True)

#     # Prepare data for LSTM
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(gold_data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD']])
#     seq_length = 60
#     x, y = create_sequences(scaled_data, seq_length)

#     # Split into training and testing datasets
#     train_size = int(len(x) * 0.8)
#     x_train, x_test = x[:train_size], x[train_size:]
#     y_train, y_test = y[:train_size], y[train_size:]

#     # Step 3: Build the LSTM model
#     model = Sequential([
#         LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
#         Dropout(0.2),
#         LSTM(64, return_sequences=False),
#         Dropout(0.2),
#         Dense(25, activation='relu'),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))

#     # Step 4: Predict and adjust prices based on sentiment
#     predicted_prices = model.predict(x_test)
#     predicted_prices = scaler.inverse_transform(
#         np.concatenate((predicted_prices, np.zeros((predicted_prices.shape[0], 4))), axis=1)
#     )[:, 0]
#     adjusted_prices = [
#         price * (1 + adjustment_factor * sentiment_score) if abs(sentiment_score) > 0.2 else price
#         for price in predicted_prices
#     ]

#     # Calculate RMSE
#     actual_prices = scaler.inverse_transform(
#         np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 4))), axis=1)
#     )[:, 0]
#     rmse = np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))
#     print(f"RMSE: {rmse}")

#     # Step 5: Predict future prices
#     future_days = 5
#     current_sequence = scaled_data[-seq_length:]
#     future_predictions = []
#     for _ in range(future_days):
#         prediction = model.predict(current_sequence[np.newaxis, :, :])[0]
#         future_predictions.append(prediction)
#         padded_prediction = np.zeros((1, current_sequence.shape[1]))
#         padded_prediction[0, 0] = prediction
#         current_sequence = np.append(current_sequence[1:], padded_prediction, axis=0)
#     future_predictions = scaler.inverse_transform(
#         np.concatenate((np.array(future_predictions), np.zeros((len(future_predictions), 4))), axis=1)
#     )[:, 0]

#     # # Step 6: Visualization
#     # plt.figure(figsize=(14, 7))
#     # plt.plot(actual_prices, label="Actual Gold Prices", color='blue')
#     # plt.plot(predicted_prices, label="Predicted Gold Prices", color='red', linestyle='dashed')
#     # plt.plot(range(len(actual_prices), len(actual_prices) + future_days), future_predictions, label="Future Predictions", color='green')
#     # plt.title("Gold Price Prediction with Sentiment Adjustment")
#     # plt.xlabel("Time")
#     # plt.ylabel("Gold Price")
#     # plt.legend()
#     # plt.grid(True)
#     # plt.show()

#     # Step 7: Save the Model
#     model.save("lstm_gold_price_model.h5")
#     print("Model saved successfully!")
#     return actual_prices, sentiment_score ,predicted_prices ,future_predictions

# # Run the program
# if __name__ == "__main__":
#     main()
