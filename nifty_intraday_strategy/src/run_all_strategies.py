"""
Run all advanced strategies automatically and find the best one.
Compares all strategies and shows which gives maximum profit.
"""

import pandas as pd
import yaml
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import time
import hashlib
import json
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_strategies import get_strategy
from backtester import Backtester
from indicators import calculate_vwap, calculate_atr
from advanced_indicators import get_all_indicators
from utils import filter_trading_hours


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


def load_strategies_config(config_path: str) -> dict:
    """Load strategies configuration from YAML file."""
    if not os.path.exists(config_path):
        # Return default config (all enabled)
        return {
            'strategies': {
                'holy_grail': {'enabled': True},
                'triangle_breakout': {'enabled': True},
                'third_candle_orb': {'enabled': True},
                'gap_open': {'enabled': True},
                'big_candle_breakout': {'enabled': True},
                'flag_breakout': {'enabled': True},
                'cpr_20ema': {'enabled': True},
                'fifteen_min_breakout': {'enabled': True},
                'last_30min_range': {'enabled': True},
                'ema20_mean_reversion': {'enabled': True},
            },
            'cache': {'enabled': True, 'cache_dir': 'cache/strategy_results', 'cache_expiry_days': 7}
        }
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_cache_key(strategy_name: str, df: pd.DataFrame, config: dict) -> str:
    """Generate cache key based on strategy, data hash, and config."""
    # Create hash of data (first and last few rows + length)
    data_hash = hashlib.md5(
        f"{len(df)}_{df['date'].iloc[0]}_{df['date'].iloc[-1]}_{df['close'].iloc[0]}_{df['close'].iloc[-1]}".encode()
    ).hexdigest()[:8]
    
    # Create hash of relevant config parameters
    config_params = {
        'atr_period': config.get('atr_period', 14),
        'atr_mult': config.get('atr_mult', 1.5),
        'risk_per_trade': config.get('risk_per_trade', 0.005),
        'initial_capital': config.get('initial_capital', 1000),
        'slippage': config.get('slippage', 0.00015),
        'commission': config.get('commission', 0.0003),
    }
    config_hash = hashlib.md5(json.dumps(config_params, sort_keys=True).encode()).hexdigest()[:8]
    
    return f"{strategy_name}_{data_hash}_{config_hash}"


def load_from_cache(cache_key: str, cache_dir: Path, expiry_days: int = 7) -> Optional[Dict]:
    """Load strategy results from cache if available and not expired."""
    cache_file = cache_dir / f"{cache_key}.json"
    
    if not cache_file.exists():
        return None
    
    # Check expiry
    if expiry_days > 0:
        file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - file_time > timedelta(days=expiry_days):
            # Cache expired, delete it
            cache_file.unlink()
            return None
    
    try:
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        
        # Convert back to proper types
        if 'trades_df' in cached_data and cached_data['trades_df']:
            cached_data['trades_df'] = pd.DataFrame(cached_data['trades_df'])
        else:
            cached_data['trades_df'] = pd.DataFrame()
        
        return cached_data
    except Exception as e:
        print(f"  ⚠ Warning: Error loading cache: {e}")
        return None


def save_to_cache(cache_key: str, result: Dict, cache_dir: Path):
    """Save strategy results to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.json"
    
    try:
        # Prepare data for JSON serialization
        cache_data = {
            'strategy_name': result['strategy_name'],
            'strategy_desc': result['strategy_desc'],
            'metrics': result['metrics'],
            'trades_df': result['trades_df'].to_dict('records') if len(result['trades_df']) > 0 else [],
            'cached_at': datetime.now().isoformat()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
        
        return True
    except Exception as e:
        print(f"  ⚠ Warning: Error saving cache: {e}")
        return False


def load_data(data_path: str, config: dict) -> pd.DataFrame:
    """Load and prepare OHLCV data."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = filter_trading_hours(df, config['start_time'], config['end_time'])
    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}\n")
    return df


def format_time(seconds):
    """Format seconds into readable time string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def print_progress(current: int, total: int, start_time: float, strategy_desc: str):
    """Print progress bar with time estimates."""
    elapsed = time.time() - start_time
    if current > 0:
        rate = elapsed / current
        remaining = (total - current) * rate
        eta = format_time(remaining)
    else:
        eta = "calculating..."
    
    elapsed_str = format_time(elapsed)
    percent = (current / total) * 100
    
    # Create progress bar (50 chars)
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    # Print progress
    print(f"\r  [{bar}] {percent:5.1f}% | {current:,}/{total:,} bars | "
          f"Elapsed: {elapsed_str} | ETA: {eta}", end='', flush=True)


def run_strategy_backtest(strategy_name: str, strategy_desc: str, df: pd.DataFrame, config: dict, 
                         cache_dir: Optional[Path] = None, cache_enabled: bool = True, 
                         expiry_days: int = 7) -> Dict:
    """Run backtest for a specific strategy and return metrics."""
    print(f"\n{'='*100}")
    print(f"Testing: {strategy_desc}")
    print(f"{'='*100}")
    
    start_time = time.time()
    
    # Generate cache key
    cache_key = None
    if cache_enabled and cache_dir:
        cache_key = get_cache_key(strategy_name, df, config)
        cached_result = load_from_cache(cache_key, cache_dir, expiry_days)
        
        if cached_result:
            print(f"  📦 Loading from cache...", end='', flush=True)
            # Reconstruct result dict
            result = {
                'strategy_name': cached_result['strategy_name'],
                'strategy_desc': cached_result['strategy_desc'],
                'metrics': cached_result['metrics'],
                'trades_df': cached_result['trades_df'],
                'backtester': None  # Not needed for cached results
            }
            print(f" ✓")
            print(f"  ✅ Loaded from cache | Trades: {len(result['trades_df'])} | PnL: ₹{result['metrics'].get('total_pnl', 0):.2f}")
            return result
    
    try:
        # Prepare data
        print("  Preparing data...", end='', flush=True)
        df_test = df.copy()
        df_test = df_test.sort_values('date').reset_index(drop=True)
        print(" ✓")
        
        # Calculate base indicators
        print("  Calculating base indicators (VWAP, ATR)...", end='', flush=True)
        df_test['vwap'] = calculate_vwap(df_test)
        df_test['atr'] = calculate_atr(df_test, config.get('atr_period', 14))
        print(" ✓")
        
        # Calculate advanced indicators
        print("  Calculating advanced indicators...", end='', flush=True)
        indicators_df = get_all_indicators(df_test, config)
        df_test = pd.concat([df_test, indicators_df], axis=1)
        print(" ✓")
        
        # Initialize backtester
        backtester = AdvancedStrategyBacktester(config, strategy_name)
        
        # Run backtest
        total_bars = len(df_test)
        current_trade = None
        prev_date = None
        
        print(f"\n  Running backtest on {total_bars:,} bars...")
        backtest_start = time.time()
        
        for i in range(len(df_test)):
            # Print progress every 1000 bars or at start/end
            if i % 1000 == 0 or i == len(df_test) - 1:
                print_progress(i + 1, total_bars, backtest_start, strategy_desc)
            row = df_test.iloc[i]
            current_time = row['date']
            current_date = current_time.date()
            
            # Check if date changed - close any open position from previous day
            if current_trade is not None and prev_date is not None and current_date != prev_date:
                prev_row = df_test.iloc[i-1]
                exit_result = backtester.execute_exit(prev_row['close'], prev_row['date'], 'End of Day')
                if exit_result:
                    trade = {
                        **current_trade,
                        **exit_result
                    }
                    trade['pnl'] = exit_result['pnl']
                    trade['return_pct'] = (exit_result['pnl'] / (current_trade['entry_price'] * current_trade['quantity'])) * 100
                    backtester.trades.append(trade)
                    current_trade = None
                    backtester.equity = backtester.cash + (backtester.position * prev_row['close'] if backtester.position > 0 else 0)
            
            # Reset daily state
            backtester.strategy.reset_daily_state(current_date)
            prev_date = current_date
            
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
                    if i + 1 < len(df_test):
                        next_bar = df_test.iloc[i + 1]
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
                entry_signal = backtester.strategy.check_entry_conditions(row, i, df_test)
                
                if entry_signal:
                    # Execute entry
                    if i + 1 < len(df_test):
                        next_bar = df_test.iloc[i + 1]
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
            last_row = df_test.iloc[-1]
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
        
        # Calculate metrics
        print("\n  Calculating metrics...", end='', flush=True)
        metrics = backtester.calculate_metrics(trades_df)
        print(" ✓")
        
        total_time = time.time() - start_time
        time_str = format_time(total_time)
        print(f"\n  ✅ Completed in {time_str} | Trades: {len(trades_df)} | PnL: ₹{metrics.get('total_pnl', 0):.2f}")
        
        result = {
            'strategy_name': strategy_name,
            'strategy_desc': strategy_desc,
            'metrics': metrics,
            'trades_df': trades_df,
            'backtester': backtester
        }
        
        # Save to cache
        if cache_enabled and cache_dir:
            print(f"  💾 Saving to cache...", end='', flush=True)
            if save_to_cache(cache_key, result, cache_dir):
                print(f" ✓")
            else:
                print(f" ✗")
        
        return result
        
    except Exception as e:
        total_time = time.time() - start_time
        time_str = format_time(total_time)
        print(f"\n  ❌ Error after {time_str}: {str(e)}")
        return None


def print_strategy_results(results: List[Dict]):
    """Print results for all strategies."""
    print("\n" + "="*100)
    print("ALL STRATEGIES RESULTS COMPARISON")
    print("="*100 + "\n")
    
    # Sort by total_pnl (descending)
    results_sorted = sorted(
        [r for r in results if r is not None],
        key=lambda x: x['metrics'].get('total_pnl', -float('inf')),
        reverse=True
    )
    
    # Print header
    print(f"{'Rank':<6} {'Strategy':<35} {'Trades':<8} {'Win Rate':<10} {'Total PnL':<15} {'Profit Factor':<15} {'Sharpe':<10} {'Max DD':<12}")
    print("-" * 100)
    
    # Print each strategy
    for rank, result in enumerate(results_sorted, 1):
        metrics = result['metrics']
        desc = result['strategy_desc']
        
        total_trades = metrics.get('total_trades', 0)
        win_rate = metrics.get('win_rate', 0)
        total_pnl = metrics.get('total_pnl', 0)
        profit_factor = metrics.get('profit_factor', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0)
        
        # Truncate description if too long
        if len(desc) > 33:
            desc = desc[:30] + "..."
        
        print(f"{rank:<6} {desc:<35} {total_trades:<8} {win_rate:>7.2f}% {total_pnl:>13.2f} ₹ {profit_factor:>13.4f} {sharpe:>8.2f} {max_dd:>10.2f}%")
    
    # Print best strategy details
    if results_sorted:
        best = results_sorted[0]
        print("\n" + "="*100)
        print("🏆 BEST STRATEGY: " + best['strategy_desc'].upper())
        print("="*100 + "\n")
        
        metrics = best['metrics']
        print(f"Strategy Name: {best['strategy_name']}")
        print(f"Description: {best['strategy_desc']}")
        print(f"\nPerformance Metrics:")
        print(f"  Total Trades: {metrics.get('total_trades', 0)}")
        print(f"  Winning Trades: {metrics.get('winning_trades', 0)}")
        print(f"  Losing Trades: {metrics.get('losing_trades', 0)}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.2f}%")
        print(f"  Total PnL: ₹{metrics.get('total_pnl', 0):.2f}")
        print(f"  Average PnL: ₹{metrics.get('avg_pnl', 0):.2f}")
        print(f"  Average Win: ₹{metrics.get('avg_win', 0):.2f}")
        print(f"  Average Loss: ₹{metrics.get('avg_loss', 0):.2f}")
        print(f"  Profit Factor: {metrics.get('profit_factor', 0):.4f}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        print(f"  Initial Capital: ₹{metrics.get('initial_capital', 0):.2f}")
        print(f"  Final Equity: ₹{metrics.get('final_equity', 0):.2f}")
        print(f"  Total Return: {metrics.get('total_return', 0):.2f}%")
        
        # Show sample trades
        if len(best['trades_df']) > 0:
            print(f"\nSample Trades (First 10):")
            print("-" * 100)
            sample_trades = best['trades_df'].head(10)
            for idx, trade in sample_trades.iterrows():
                print(f"  {trade['side']:>5} | Entry: {trade['entry_time']} @ ₹{trade['entry_price']:.2f} | "
                      f"Exit: {trade['exit_time']} @ ₹{trade['exit_price']:.2f} | "
                      f"PnL: ₹{trade['pnl']:.2f} ({trade['return_pct']:.2f}%) | Reason: {trade['exit_reason']}")
        
        return best
    else:
        print("\nNo successful strategy results!")
        return None


def save_best_strategy(best_result: Dict, project_root: Path):
    """Save best strategy results."""
    if best_result is None:
        return
    
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    strategy_name = best_result['strategy_name']
    
    # Save trades
    if len(best_result['trades_df']) > 0:
        trades_file = results_dir / f'BEST_{strategy_name}_trades.csv'
        best_result['trades_df'].to_csv(trades_file, index=False)
        print(f"\nBest strategy trades saved to: {trades_file}")
    
    # Save equity curve
    equity_curve_df = pd.DataFrame(best_result['backtester'].equity_curve)
    equity_file = results_dir / f'BEST_{strategy_name}_equity_curve.csv'
    equity_curve_df.to_csv(equity_file, index=False)
    print(f"Best strategy equity curve saved to: {equity_file}")
    
    # Save metrics
    import json
    metrics_file = results_dir / f'BEST_{strategy_name}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(best_result['metrics'], f, indent=2)
    print(f"Best strategy metrics saved to: {metrics_file}")


def main():
    """Main function to run all strategies and find the best one."""
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
    
    # Load strategies configuration
    strategies_config_path = project_root / 'strategies_config.yaml'
    print("Loading strategies configuration...")
    strategies_config = load_strategies_config(str(strategies_config_path))
    
    # Disable plots
    config['generate_plots'] = False
    
    # Load data
    df = load_data(str(data_path), config)
    
    # Get cache settings
    cache_enabled = strategies_config.get('cache', {}).get('enabled', True)
    cache_dir_name = strategies_config.get('cache', {}).get('cache_dir', 'cache/strategy_results')
    cache_dir = project_root / cache_dir_name
    expiry_days = strategies_config.get('cache', {}).get('cache_expiry_days', 7)
    
    if cache_enabled:
        print(f"Cache enabled: {cache_dir}")
        print(f"Cache expiry: {expiry_days} days\n")
    else:
        print("Cache disabled\n")
    
    # All strategies with their config
    all_strategies = {
        'holy_grail': 'Holy Grail Strategy (EMA pullback in strong trend)',
        'triangle_breakout': 'Triangle Breakout with Confirmation (5m only)',
        'third_candle_orb': 'Range Breakout of 3rd 5-min Candle',
        'gap_open': 'Gap Open Strategy (2nd 5m candle)',
        'big_candle_breakout': 'Big Candle Breakout Strategy',
        'flag_breakout': 'Flag Breakout',
        'cpr_20ema': 'CPR + 20 EMA Strategy',
        'fifteen_min_breakout': '15-Minute Breakout Strategy',
        'last_30min_range': 'Last 30-min Range of Previous Day',
        'ema20_mean_reversion': '20 EMA Mean Reversion (range days only)',
    }
    
    # Filter to only enabled strategies
    strategies = []
    for strategy_id, strategy_desc in all_strategies.items():
        strategy_config = strategies_config.get('strategies', {}).get(strategy_id, {})
        if strategy_config.get('enabled', True):
            strategies.append((strategy_id, strategy_desc))
        else:
            print(f"⏭️  Skipping {strategy_desc} (disabled in strategies_config.yaml)")
    
    if not strategies:
        print("\n❌ No strategies enabled! Please enable at least one strategy in strategies_config.yaml")
        return
    
    print("="*100)
    print("RUNNING ALL STRATEGIES AUTOMATICALLY")
    print("="*100)
    print(f"\nTesting {len(strategies)} strategies...")
    print(f"Total bars to process: {len(df):,}")
    print(f"Estimated time per strategy: 10-30 minutes (depending on strategy complexity)")
    print(f"Total estimated time: {len(strategies) * 20} minutes ({len(strategies) * 20 / 60:.1f} hours)\n")
    
    overall_start = time.time()
    
    # Run all strategies
    results = []
    for idx, (strategy_id, strategy_desc) in enumerate(strategies, 1):
        print(f"\n{'#'*100}")
        print(f"STRATEGY {idx}/{len(strategies)}")
        print(f"{'#'*100}")
        
        result = run_strategy_backtest(strategy_id, strategy_desc, df, config, 
                                     cache_dir, cache_enabled, expiry_days)
        if result:
            results.append(result)
        
        # Show overall progress
        elapsed_total = time.time() - overall_start
        avg_time_per_strategy = elapsed_total / idx
        remaining_strategies = len(strategies) - idx
        estimated_remaining = avg_time_per_strategy * remaining_strategies
        
        print(f"\n  Overall Progress: {idx}/{len(strategies)} strategies completed")
        print(f"  Time elapsed: {format_time(elapsed_total)}")
        print(f"  Estimated remaining: {format_time(estimated_remaining)}")
        print(f"  Average time per strategy: {format_time(avg_time_per_strategy)}")
    
    # Print comparison results
    best_result = print_strategy_results(results)
    
    # Save best strategy
    if best_result:
        save_best_strategy(best_result, project_root)
        print("\n✅ Analysis complete! Best strategy identified and saved.")
    else:
        print("\n❌ No profitable strategies found.")


if __name__ == '__main__':
    main()

