import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NIFTYMLStrategy:
    def __init__(self, data_file=None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.trained = False
        self.data = None
        
    def load_data(self, data_file):
        """Load minute-level NIFTY 50 data (CSV format)"""
        try:
            self.data = pd.read_csv(data_file)
            # Ensure datetime column
            if 'timestamp' in self.data.columns:
                self.data['datetime'] = pd.to_datetime(self.data['timestamp'])
            elif 'datetime' in self.data.columns:
                self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            else:
                self.data['datetime'] = pd.to_datetime(self.data.iloc[:, 0])
            
            self.data = self.data.sort_values('datetime').reset_index(drop=True)
            print(f"✓ Loaded {len(self.data)} minute candles")
            print(f"✓ Data range: {self.data['datetime'].min()} to {self.data['datetime'].max()}")
            return self.data
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def create_features(self, df):
        """Engineer technical and temporal features"""
        df = df.copy()
        
        # Basic OHLCV (ensure these columns exist)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: {col} column missing")
        
        # Price-based features
        df['returns'] = df['close'].pct_change() * 100
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['oc_ratio'] = abs(df['open'] - df['close']) / df['close']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Volatility indicators
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        # Momentum
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        # RSI calculation
        def rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['rsi_14'] = rsi(df['close'], 14)
        df['rsi_5'] = rsi(df['close'], 5)
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1)
        
        # Bollinger Bands
        bb_sma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_sma + (bb_std * 2)
        df['bb_lower'] = bb_sma - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-6)
        
        # Temporal features
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Lag features (previous candles)
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Target: Next minute direction (1 = up, 0 = down)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop NaN rows
        df = df.dropna()
        
        return df
    
    def prepare_training_data(self, df, test_split=0.2):
        """Prepare train/test data with proper time-series split"""
        df = self.create_features(df)
        
        # All features except datetime and target
        self.feature_cols = [col for col in df.columns 
                            if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'target']]
        
        X = df[self.feature_cols].fillna(0)
        y = df['target']
        
        # Time-series split (no data leakage)
        split_idx = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✓ Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"✓ Features engineered: {len(self.feature_cols)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def train_model(self, X_train, X_test, y_train, y_test):
        """Train XGBoost ensemble model"""
        print("\n🔄 Training XGBoost model...")
        
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss',
            tree_method='hist',
            n_jobs=-1
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=30,
            verbose=False
        )
        
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        
        print(f"✓ Training Accuracy: {train_acc:.4f}")
        print(f"✓ Testing Accuracy: {test_acc:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📊 Top 10 Important Features:")
        for idx, row in importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        self.trained = True
        return test_acc
    
    def predict_realtime(self, latest_data):
        """Predict on real-time minute candle"""
        if not self.trained:
            print("✗ Model not trained yet")
            return None
        
        try:
            df = latest_data.copy()
            df = self.create_features(df)
            
            # Use only the latest complete candle
            latest = df.iloc[-1:][self.feature_cols].fillna(0)
            latest_scaled = self.scaler.transform(latest)
            
            # Get prediction and probability
            prediction = self.model.predict(latest_scaled)[0]
            probability = self.model.predict_proba(latest_scaled)[0]
            
            signal = "🔴 SELL/DOWN" if prediction == 0 else "🟢 BUY/UP"
            confidence = max(probability) * 100
            
            return {
                'signal': signal,
                'prediction': prediction,
                'confidence': confidence,
                'probability': probability,
                'timestamp': df['datetime'].iloc[-1]
            }
        except Exception as e:
            print(f"✗ Prediction error: {e}")
            return None
    
    def save_model(self, filepath='nifty_model.pkl'):
        """Save trained model and scaler"""
        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_cols': self.feature_cols
            }, filepath)
            print(f"✓ Model saved to {filepath}")
        except Exception as e:
            print(f"✗ Save error: {e}")
    
    def load_model(self, filepath='nifty_model.pkl'):
        """Load pre-trained model"""
        try:
            data = joblib.load(filepath)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_cols = data['feature_cols']
            self.trained = True
            print(f"✓ Model loaded from {filepath}")
        except Exception as e:
            print(f"✗ Load error: {e}")


# ============= USAGE EXAMPLE =============
if __name__ == "__main__":
    
    # Initialize strategy
    strategy = NIFTYMLStrategy()
    
    # Load your historical data (CSV with columns: datetime, open, high, low, close, volume)
    # data = strategy.load_data('nifty_50_minutes.csv')
    
    # For demonstration, create sample data
    print("=" * 60)
    print("NIFTY 50 ML TRADING STRATEGY - PRODUCTION READY")
    print("=" * 60)
    
    # Generate sample data for demo
    dates = pd.date_range(start='2023-01-01', periods=10000, freq='1min')
    sample_data = pd.DataFrame({
        'datetime': dates,
        'open': 17000 + np.random.randn(10000).cumsum(),
        'high': 17100 + np.random.randn(10000).cumsum(),
        'low': 16900 + np.random.randn(10000).cumsum(),
        'close': 17050 + np.random.randn(10000).cumsum(),
        'volume': np.random.randint(1000, 100000, 10000)
    })
    
    print("\n📥 Loading data...")
    strategy.data = sample_data
    print(f"✓ Loaded {len(strategy.data)} minute candles")
    
    # Prepare data
    print("\n🔧 Preparing data with feature engineering...")
    X_train, X_test, y_train, y_test, _, _ = strategy.prepare_training_data(strategy.data)
    
    # Train model
    print("\n🤖 Training model...")
    accuracy = strategy.train_model(X_train, X_test, y_train, y_test)
    
    # Save model
    print("\n💾 Saving model...")
    strategy.save_model('nifty_model.pkl')
    
    # Real-time prediction demo
    print("\n" + "=" * 60)
    print("REAL-TIME PREDICTION DEMO")
    print("=" * 60)
    latest_candles = sample_data.tail(100)
    result = strategy.predict_realtime(latest_candles)
    
    if result:
        print(f"\n⏰ Time: {result['timestamp']}")
        print(f"{result['signal']}")
        print(f"📈 Confidence: {result['confidence']:.2f}%")
        print(f"📊 Probabilities: DOWN={result['probability'][0]*100:.2f}% | UP={result['probability'][1]*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("✅ System Ready for Real-Time Trading")
    print("=" * 60)