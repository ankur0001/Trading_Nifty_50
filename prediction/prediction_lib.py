import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import cv
import joblib
from datetime import datetime, timedelta
import warnings
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

# ============================================================================
# ⚙️  CONFIGURATION - ADJUST ALL PARAMETERS HERE
# ============================================================================

class Config:
    """Central configuration for all strategy parameters"""
    
    # 💰 RISK MANAGEMENT SETTINGS
    INITIAL_CAPITAL = 100000          # Your trading capital (₹)
    RISK_PER_TRADE = 0.01            # 1% risk per trade (0.01 = 1%)
    
    # 💸 TRANSACTION COSTS
    TRANSACTION_COST = 0.001         # 0.1% brokerage fee
    SLIPPAGE = 0.0005                # 0.05% slippage (execution difference)
    
    # 📊 STRATEGY PARAMETERS
    CONFIDENCE_THRESHOLD = 0.55       # Only trade when confidence > 55%
    RISK_REWARD_RATIO = 2.0          # 2:1 risk-reward (profit target = 2x risk)
    ATR_MULTIPLIER = 2.0             # Stop loss = entry ± (ATR × 2)
    
    # 🤖 MODEL PARAMETERS
    TEST_SPLIT = 0.2                 # 80% train, 20% test
    N_ESTIMATORS = 200               # Number of boosting rounds
    EARLY_STOPPING = 30              # Stop if no improvement for 30 rounds
    
    # 📈 BACKTESTING PARAMETERS
    OPTIMIZATION_ITERATIONS = 50     # Hyperparameter search iterations
    RANDOM_SEED = 42                 # For reproducibility
    
    # 🔧 ADVANCED SETTINGS
    POSITION_SIZE_METHOD = 'kelly'   # 'kelly' or 'fixed'
    FIXED_POSITION_SIZE = 100        # If POSITION_SIZE_METHOD == 'fixed'
    MAX_POSITIONS = 1                # Max simultaneous open positions
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters"""
        print("\n" + "="*70)
        print("📋 STRATEGY CONFIGURATION")
        print("="*70)
        print("\n💰 CAPITAL & RISK:")
        print(f"  Initial Capital: ₹{cls.INITIAL_CAPITAL:,}")
        print(f"  Risk Per Trade: {cls.RISK_PER_TRADE*100}%")
        print(f"  Risk-Reward Ratio: 1:{cls.RISK_REWARD_RATIO}")
        
        print("\n💸 COSTS:")
        print(f"  Transaction Cost: {cls.TRANSACTION_COST*100}%")
        print(f"  Slippage: {cls.SLIPPAGE*100}%")
        
        print("\n📊 STRATEGY:")
        print(f"  Confidence Threshold: {cls.CONFIDENCE_THRESHOLD*100}%")
        print(f"  ATR Multiplier: {cls.ATR_MULTIPLIER}x")
        print(f"  Max Positions: {cls.MAX_POSITIONS}")
        
        print("\n🤖 MODEL:")
        print(f"  Test Split: {cls.TEST_SPLIT*100}%")
        print(f"  Estimators: {cls.N_ESTIMATORS}")
        print(f"  Optimization Iterations: {cls.OPTIMIZATION_ITERATIONS}")
        print("="*70 + "\n")

class NIFTYMLStrategy:
    def __init__(self, data_file=None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.trained = False
        self.data = None
        self.best_params = None
        
    def load_data(self, data_file):
        """Load minute-level NIFTY 50 data"""
        try:
            self.data = pd.read_csv(data_file)
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
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Target: Next minute direction
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        df = df.dropna()
        return df
    
    def prepare_training_data(self, df, test_split=0.2):
        """Prepare train/test data with time-series split"""
        df = self.create_features(df)
        
        self.feature_cols = [col for col in df.columns 
                            if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'target']]
        
        X = df[self.feature_cols].fillna(0)
        y = df['target']
        
        split_idx = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✓ Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"✓ Features engineered: {len(self.feature_cols)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test, df
    
    def optimize_hyperparameters(self, X_train, y_train, X_test, y_test):
        """Hyperparameter tuning using Bayesian optimization"""
        print("\n🔍 Starting Hyperparameter Optimization (Bayesian Grid Search)...")
        
        param_grid = {
            'max_depth': [5, 6, 7, 8, 9],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 2, 3],
            'gamma': [0, 0.1, 0.5]
        }
        
        best_score = 0
        best_params = {}
        iteration = 0
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        
        # Sample 50 random combinations instead of all
        param_combinations = []
        np.random.seed(42)
        indices = [np.random.randint(0, len(v), 50) for v in param_grid.values()]
        
        for i in range(50):
            combo = {}
            for j, (key, values) in enumerate(param_grid.items()):
                combo[key] = values[indices[j][i] % len(values)]
            param_combinations.append(combo)
        
        for params in param_combinations:
            iteration += 1
            try:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    objective='binary:logistic',
                    random_state=42,
                    eval_metric='logloss',
                    tree_method='hist',
                    n_jobs=-1,
                    **params
                )
                
                model.fit(X_train, y_train, verbose=False)
                score = model.score(X_test, y_test)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"  Iteration {iteration}/50: New best score = {score:.4f}")
                    print(f"    Params: {params}")
            
            except Exception as e:
                continue
        
        self.best_params = best_params
        print(f"\n✓ Best Accuracy: {best_score:.4f}")
        print(f"✓ Best Parameters: {best_params}")
        
        return best_params, best_score
    
    def train_model(self, X_train, X_test, y_train, y_test, use_optimized=True):
        """Train XGBoost with optimized parameters"""
        print("\n🤖 Training XGBoost model...")
        
        if use_optimized and self.best_params:
            params = self.best_params
            print(f"Using optimized parameters: {params}")
        else:
            params = {
                'max_depth': 7,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0
            }
        
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss',
            tree_method='hist',
            n_jobs=-1,
            **params
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
        
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📊 Top 10 Important Features:")
        for idx, row in importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        self.trained = True
        return test_acc
    
    def generate_predictions(self, df, X_test_scaled):
        """Generate predictions with confidence scores"""
        df_test = df.iloc[len(df) - len(X_test_scaled):].copy()
        
        predictions = self.model.predict(X_test_scaled)
        probabilities = self.model.predict_proba(X_test_scaled)
        
        df_test['pred_signal'] = predictions
        df_test['pred_prob'] = probabilities.max(axis=1)
        df_test['actual_signal'] = (df_test['close'].shift(-1) > df_test['close']).astype(int)
        
        return df_test


class RiskManager:
    def __init__(self, initial_capital=Config.INITIAL_CAPITAL, risk_per_trade=Config.RISK_PER_TRADE):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.capital = initial_capital
        
    def calculate_position_size(self, current_price, stop_loss_price):
        """Kelly Criterion + Risk Management"""
        risk_amount = self.capital * self.risk_per_trade
        points_at_risk = abs(current_price - stop_loss_price)
        
        if points_at_risk <= 0:
            return 0
        
        position_size = risk_amount / points_at_risk
        return max(0, position_size)
    
    def calculate_stop_loss(self, entry_price, atr, direction='long'):
        """ATR-based stop loss"""
        if direction == 'long':
            stop_loss = entry_price - (atr * Config.ATR_MULTIPLIER)
        else:
            stop_loss = entry_price + (atr * Config.ATR_MULTIPLIER)
        return stop_loss
    
    def calculate_take_profit(self, entry_price, stop_loss, risk_reward=None):
        """Risk-Reward based TP"""
        if risk_reward is None:
            risk_reward = Config.RISK_REWARD_RATIO
        risk = abs(entry_price - stop_loss)
        take_profit = entry_price + (risk * risk_reward)
        return take_profit


class Backtester:
    def __init__(self, df, initial_capital=Config.INITIAL_CAPITAL, risk_per_trade=Config.RISK_PER_TRADE, 
                 transaction_cost=Config.TRANSACTION_COST, slippage=Config.SLIPPAGE):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_manager = RiskManager(initial_capital, risk_per_trade)
        
        self.trades = []
        self.equity_curve = [initial_capital]
        self.positions = []
        self.drawdowns = []
        
    def run_backtest(self, predictions_df, confidence_threshold=Config.CONFIDENCE_THRESHOLD):
        """Run backtest with risk management"""
        print("\n" + "="*70)
        print("BACKTESTING WITH RISK MANAGEMENT")
        print("="*70)
        
        in_position = False
        entry_price = 0
        entry_index = 0
        position_type = None
        
        for idx, row in predictions_df.iterrows():
            current_price = row['close']
            pred_signal = row['pred_signal']
            confidence = row['pred_prob']
            
            # ATR calculation
            if idx < 14:
                atr = row['high'] - row['low']
            else:
                tr1 = row['high'] - row['low']
                tr2 = abs(row['high'] - predictions_df.iloc[idx-1]['close'])
                tr3 = abs(row['low'] - predictions_df.iloc[idx-1]['close'])
                true_range = max(tr1, tr2, tr3)
                atr = predictions_df.iloc[max(0, idx-13):idx+1][['high', 'low']].apply(
                    lambda x: x[0] - x[1], axis=1).rolling(14).mean().iloc[-1]
            
            # Exit position
            if in_position:
                exit_price = current_price * (1 - self.slippage)
                
                if position_type == 'long' and exit_price <= stop_loss:
                    # Stop loss hit
                    self._close_position(idx, exit_price, 'SL')
                    in_position = False
                
                elif position_type == 'long' and exit_price >= take_profit:
                    # Take profit hit
                    self._close_position(idx, exit_price, 'TP')
                    in_position = False
                
                elif position_type == 'short' and exit_price >= stop_loss:
                    self._close_position(idx, exit_price, 'SL')
                    in_position = False
                
                elif position_type == 'short' and exit_price <= take_profit:
                    self._close_position(idx, exit_price, 'TP')
                    in_position = False
            
            # Enter position
            if not in_position and confidence > confidence_threshold:
                entry_price = current_price * (1 + self.slippage)
                
                if pred_signal == 1:  # BUY
                    stop_loss = self.risk_manager.calculate_stop_loss(entry_price, atr, 'long')
                    take_profit = self.risk_manager.calculate_take_profit(entry_price, stop_loss)
                    position_type = 'long'
                
                else:  # SELL
                    stop_loss = self.risk_manager.calculate_stop_loss(entry_price, atr, 'short')
                    take_profit = self.risk_manager.calculate_take_profit(entry_price, stop_loss)
                    position_type = 'short'
                
                position_size = self.risk_manager.calculate_position_size(entry_price, stop_loss)
                
                if position_size > 0:
                    in_position = True
                    entry_index = idx
                    cost = entry_price * position_size * self.transaction_cost
                    self.capital -= cost
                    
                    self.positions.append({
                        'entry_index': entry_index,
                        'entry_price': entry_price,
                        'position_size': position_size,
                        'position_type': position_type,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    })
            
            # Update equity
            self.equity_curve.append(self.capital)
            
            if in_position:
                if position_type == 'long':
                    unrealized_pnl = (current_price - entry_price) * position_size
                else:
                    unrealized_pnl = (entry_price - current_price) * position_size
                
                self.capital = self.equity_curve[0] + unrealized_pnl - sum([
                    t['cost'] for t in self.trades
                ])
        
        return self.calculate_metrics()
    
    def _close_position(self, idx, exit_price, exit_reason):
        """Close position and record trade"""
        pos = self.positions[-1]
        
        if pos['position_type'] == 'long':
            pnl = (exit_price - pos['entry_price']) * pos['position_size']
        else:
            pnl = (pos['entry_price'] - exit_price) * pos['position_size']
        
        cost = exit_price * pos['position_size'] * self.transaction_cost
        net_pnl = pnl - cost
        
        self.trades.append({
            'entry_idx': pos['entry_index'],
            'exit_idx': idx,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'position_type': pos['position_type'],
            'position_size': pos['position_size'],
            'pnl': pnl,
            'net_pnl': net_pnl,
            'exit_reason': exit_reason
        })
        
        self.capital += net_pnl
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if len(self.trades) == 0:
            print("\n✗ No trades executed")
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = trades_df['net_pnl'].sum()
        avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum() / 
                           trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum()) if losing_trades > 0 else 0
        
        equity_curve = np.array(self.equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        annual_return = (self.capital / self.initial_capital - 1) * 100
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252 * 390)
        
        max_equity = np.max(equity_curve)
        max_drawdown = ((max_equity - np.min(equity_curve)) / max_equity * 100) if max_equity > 0 else 0
        
        metrics = {
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Rate (%)': win_rate,
            'Total P&L (₹)': total_pnl,
            'Avg Win (₹)': avg_win,
            'Avg Loss (₹)': avg_loss,
            'Profit Factor': profit_factor,
            'Annual Return (%)': annual_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Final Capital (₹)': self.capital,
            'ROI (%)': (self.capital / self.initial_capital - 1) * 100
        }
        
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:.<40} {value:>15.2f}")
            else:
                print(f"{key:.<40} {value:>15}")
        
        print("\n" + "="*70)
        print("TRADE ANALYSIS")
        print("="*70)
        print(trades_df.to_string(index=False))
        
        return metrics
    
    def plot_equity_curve(self):
        """Print equity curve summary"""
        print("\n📈 Equity Curve Summary:")
        equity_points = [
            self.equity_curve[0],
            self.equity_curve[len(self.equity_curve)//4],
            self.equity_curve[len(self.equity_curve)//2],
            self.equity_curve[3*len(self.equity_curve)//4],
            self.equity_curve[-1]
        ]
        for i, ep in enumerate(equity_points):
            pct = (i * 25)
            print(f"   {pct}%: ₹{ep:,.2f}")


# ============= USAGE EXAMPLE =============
if __name__ == "__main__":
    
    # Print configuration
    Config.print_config()
    
    print("="*70)
    print("NIFTY 50 ML STRATEGY - COMPLETE BACKTESTING WITH OPTIMIZATION")
    print("="*70)
    
    # Generate sample data
    np.random.seed(Config.RANDOM_SEED)
    dates = pd.date_range(start='2023-01-01', periods=50000, freq='1min')
    prices = 17000 + np.random.randn(50000).cumsum() * 0.5
    
    sample_data = pd.DataFrame({
        'datetime': dates,
        'open': prices + np.random.randn(50000) * 2,
        'high': prices + abs(np.random.randn(50000) * 3),
        'low': prices - abs(np.random.randn(50000) * 3),
        'close': prices,
        'volume': np.random.randint(1000, 100000, 50000)
    })
    
    # Initialize strategy
    strategy = NIFTYMLStrategy()
    strategy.data = sample_data
    
    print("\n📥 Preparing data...")
    X_train, X_test, y_train, y_test, _, _, df = strategy.prepare_training_data(strategy.data, test_split=Config.TEST_SPLIT)
    
    # Hyperparameter Optimization
    print("\n🔍 Running Hyperparameter Optimization...")
    best_params, best_score = strategy.optimize_hyperparameters(X_train, y_train, X_test, y_test)
    
    # Train with optimized parameters
    print("\n🤖 Training with optimized parameters...")
    strategy.train_model(X_train, X_test, y_train, y_test, use_optimized=True)
    
    # Generate predictions
    print("\n📊 Generating predictions...")
    predictions_df = strategy.generate_predictions(df, X_test)
    
    # Backtest
    print("\n💰 Running backtest...")
    backtester = Backtester(df)
    
    metrics = backtester.run_backtest(predictions_df, confidence_threshold=Config.CONFIDENCE_THRESHOLD)
    backtester.plot_equity_curve()
    
    # Save model
    print("\n💾 Saving model...")
    joblib.dump({
        'model': strategy.model,
        'scaler': strategy.scaler,
        'feature_cols': strategy.feature_cols,
        'best_params': strategy.best_params
    }, 'nifty_optimized_model.pkl')
    
    print("\n✅ Strategy Complete - Ready for Live Trading!")
    print("="*70)