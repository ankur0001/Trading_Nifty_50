from prediction_lib.py import NIFTYMLStrategy 
from prediction_lib.py import Backtester 

# 1. Load data
strategy = NIFTYMLStrategy()
X_train, X_test, y_train, y_test, _, _, df = strategy.prepare_training_data(your_data)

# 2. Optimize hyperparameters (takes 2-3 mins)
best_params, score = strategy.optimize_hyperparameters(X_train, y_train, X_test, y_test)

# 3. Train with best params
strategy.train_model(X_train, X_test, y_train, y_test, use_optimized=True)

# 4. Generate predictions
predictions_df = strategy.generate_predictions(df, X_test)

# 5. Run backtest
backtester = Backtester(df, initial_capital=100000, risk_per_trade=0.01)
metrics = backtester.run_backtest(predictions_df, confidence_threshold=0.55)

# 6. Print results
backtester.plot_equity_curve()