# 1. Load your 10-year minute data (CSV format)
strategy = NIFTYMLStrategy()
data = strategy.load_data('data/nifty_1min.csv')

# 2. Train the model
X_train, X_test, y_train, y_test, _, _ = strategy.prepare_training_data(data)
strategy.train_model(X_train, X_test, y_train, y_test)

# 3. Save for production
strategy.save_model('nifty_model.pkl')

# 4. Real-time prediction (on new minute candle)
latest_100_candles = df.tail(100)  # Get last 100 minutes
result = strategy.predict_realtime(latest_100_candles)
print(result['signal'])  # 🟢 BUY/UP or 🔴 SELL/DOWN
print(f"Confidence: {result['confidence']:.2f}%")