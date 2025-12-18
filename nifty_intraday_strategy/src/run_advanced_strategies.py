"""
Run backtests for advanced strategies separately.
This allows testing each strategy independently.
"""

import pandas as pd
import yaml
import sys
import os
from pathlib import Path
from typing import Dict

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_strategies import get_strategy
from backtester import Backtester
from indicators import calculate_vwap, calculate_atr
from advanced_indicators import get_all_indicators
from utils import filter_trading_hours, plot_equity_curve, plot_pnl_histogram, print_backtest_summary


class AdvancedStrategyBacktester(Backtester):
    """Backtester for advanced strategies."""
    
    def __init__(self, config: Dict, strategy_name: str):
        """Initialize with specific strategy."""
        # Initialize base backtester
        self.initial_capital = config['initial_capital']
        self.risk_per_trade = config['risk_per_trade']
        self.slippage = config['slippage']
        self.commission = config['commission']
        
        # Use advanced strategy
        self.strategy = get_strategy(strategy_name, config)
        
        # Track state
        self.equity = self.initial_capital
        self.cash = self.initial_capital
        self.position = 0
        self.position_side = None
        self.entry_price = None
        self.entry_time = None
        
        # Results
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_path: str, config: dict) -> pd.DataFrame:
    """Load and prepare OHLCV data."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = filter_trading_hours(df, config['start_time'], config['end_time'])
    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    return df


def run_strategy_backtest(strategy_name: str, df: pd.DataFrame, config: dict):
    """Run backtest for a specific strategy."""
    print("\n" + "="*70)
    print(f"RUNNING BACKTEST: {strategy_name.upper().replace('_', ' ')}")
    print("="*70 + "\n")
    
    # Prepare data
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate base indicators
    print("Calculating base indicators (VWAP, ATR)...")
    df['vwap'] = calculate_vwap(df)
    df['atr'] = calculate_atr(df, config.get('atr_period', 14))
    
    # Calculate advanced indicators (needed for strategies)
    print("Calculating advanced indicators...")
    indicators_df = get_all_indicators(df, config)
    df = pd.concat([df, indicators_df], axis=1)
    
    # Initialize backtester
    backtester = AdvancedStrategyBacktester(config, strategy_name)
    
    # Run backtest
    print("Running backtest...")
    total_bars = len(df)
    current_trade = None
    
    for i in range(len(df)):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{total_bars} bars ({100*(i+1)/total_bars:.1f}%)")
        
        row = df.iloc[i]
        current_time = row['date']
        current_date = current_time.date()
        
        # Reset daily state
        backtester.strategy.reset_daily_state(current_date)
        
        # Skip if missing indicators
        if pd.isna(row['vwap']) or pd.isna(row['atr']):
            backtester.equity_curve.append({
                'date': current_time,
                'equity': backtester.equity,
                'cash': backtester.cash,
                'position_value': backtester.position * row['close'] if backtester.position > 0 else 0
            })
            continue
        
        # Check for exit first
        if current_trade is not None:
            should_exit = backtester.strategy.check_exit_conditions(
                row, row['vwap'], current_time
            )
            
            if should_exit:
                # Determine exit price and reason
                if backtester.position_side == 'LONG':
                    if row['low'] <= backtester.strategy.stop_loss:
                        exit_price = backtester.strategy.stop_loss
                        reason = 'Stop Loss'
                    else:
                        exit_price = row['close']
                        reason = 'Exit Signal'
                else:  # SHORT
                    if row['high'] >= backtester.strategy.stop_loss:
                        exit_price = backtester.strategy.stop_loss
                        reason = 'Stop Loss'
                    else:
                        exit_price = row['close']
                        reason = 'Exit Signal'
                
                # Execute exit
                if i + 1 < len(df):
                    next_bar = df.iloc[i + 1]
                    exit_result = backtester.execute_exit(next_bar['open'], next_bar['date'], reason)
                else:
                    exit_result = backtester.execute_exit(row['close'], current_time, reason)
                
                if exit_result:
                    trade = {
                        **current_trade,
                        **exit_result
                    }
                    trade['pnl'] = exit_result['pnl']
                    trade['return_pct'] = (exit_result['pnl'] / (current_trade['entry_price'] * current_trade['quantity'])) * 100
                    backtester.trades.append(trade)
                    
                    backtester.equity = backtester.cash + (backtester.position * row['close'] if backtester.position > 0 else 0)
                    current_trade = None
        
        # Check for entry
        if current_trade is None and backtester.position == 0:
            entry_signal = backtester.strategy.check_entry_conditions(row, i, df)
            
            if entry_signal:
                # Execute entry
                if i + 1 < len(df):
                    next_bar = df.iloc[i + 1]
                    entry_result = backtester.execute_entry(
                        entry_signal, next_bar['open'], next_bar['date'], row['atr']
                    )
                else:
                    entry_result = backtester.execute_entry(
                        entry_signal, row['close'], current_time, row['atr']
                    )
                
                if entry_result:
                    current_trade = entry_result
                    backtester.equity = backtester.cash + (backtester.position * row['close'] if backtester.position > 0 else 0)
        
        # Update equity curve
        position_value = backtester.position * row['close'] if backtester.position > 0 else 0
        backtester.equity = backtester.cash + position_value
        
        backtester.equity_curve.append({
            'date': current_time,
            'equity': backtester.equity,
            'cash': backtester.cash,
            'position_value': position_value
        })
    
    # Close any open position
    if current_trade is not None:
        last_row = df.iloc[-1]
        exit_result = backtester.execute_exit(last_row['close'], last_row['date'], 'End of Data')
        if exit_result:
            trade = {
                **current_trade,
                **exit_result
            }
            trade['pnl'] = exit_result['pnl']
            trade['return_pct'] = (exit_result['pnl'] / (current_trade['entry_price'] * current_trade['quantity'])) * 100
            backtester.trades.append(trade)
    
    # Convert trades to DataFrame
    if backtester.trades:
        trades_df = pd.DataFrame(backtester.trades)
    else:
        trades_df = pd.DataFrame()
    
    return trades_df, backtester


def main():
    """Main function to run strategy backtests."""
    # Get project root
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config.yaml'
    data_path = project_root / 'data' / 'nifty_1min.csv'
    
    # Check files
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(str(config_path))
    
    # Load data
    df = load_data(str(data_path), config)
    
    # Available strategies
    strategies = {
        '1': ('holy_grail', 'Holy Grail Strategy (EMA pullback in strong trend)'),
        '2': ('triangle_breakout', 'Triangle Breakout with Confirmation (5m only)'),
        '3': ('third_candle_orb', 'Range Breakout of 3rd 5-min Candle'),
        '4': ('gap_open', 'Gap Open Strategy (2nd 5m candle)'),
        '5': ('big_candle_breakout', 'Big Candle Breakout Strategy'),
        '6': ('flag_breakout', 'Flag Breakout'),
        '7': ('cpr_20ema', 'CPR + 20 EMA Strategy'),
        '8': ('fifteen_min_breakout', '15-Minute Breakout Strategy'),
        '9': ('last_30min_range', 'Last 30-min Range of Previous Day'),
        '10': ('ema20_mean_reversion', '20 EMA Mean Reversion (range days only)'),
    }
    
    # Ask user to select strategy
    print("\n" + "="*70)
    print("ADVANCED STRATEGIES BACKTEST")
    print("="*70)
    print("\nAvailable strategies:")
    for key, (strategy_id, description) in strategies.items():
        print(f"  {key}. {description}")
    
    choice = input("\nEnter strategy number (1-10): ").strip()
    
    if choice not in strategies:
        print(f"Invalid choice: {choice}")
        return
    
    strategy_id, strategy_desc = strategies[choice]
    
    # Run backtest
    trades_df, backtester = run_strategy_backtest(strategy_id, df, config)
    
    # Calculate metrics
    print("\nCalculating performance metrics...")
    metrics = backtester.calculate_metrics(trades_df)
    
    # Print summary
    print_backtest_summary(metrics, trades_df)
    
    # Generate plots
    if config.get('generate_plots', True):
        print("Generating plots...")
        equity_curve_df = pd.DataFrame(backtester.equity_curve)
        
        if len(equity_curve_df) > 0:
            plot_equity_curve(equity_curve_df)
        
        if len(trades_df) > 0:
            plot_pnl_histogram(trades_df)
    
    # Save results
    if config.get('save_results', True):
        results_dir = project_root / 'results'
        results_dir.mkdir(exist_ok=True)
        
        equity_curve_df = pd.DataFrame(backtester.equity_curve)
        
        if len(trades_df) > 0:
            output_file = results_dir / f'{strategy_id}_trades.csv'
            trades_df.to_csv(output_file, index=False)
            print(f"\nTrades saved to {output_file}")
        
        equity_file = results_dir / f'{strategy_id}_equity_curve.csv'
        equity_curve_df.to_csv(equity_file, index=False)
        print(f"Equity curve saved to {equity_file}")
        
        # Save metrics
        import json
        metrics_file = results_dir / f'{strategy_id}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_file}")
    
    print(f"\n✅ Backtest completed for: {strategy_desc}")


if __name__ == '__main__':
    main()

