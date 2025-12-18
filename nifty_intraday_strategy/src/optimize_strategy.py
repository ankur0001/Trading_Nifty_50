"""
Strategy optimization script to find best parameters for maximum profit.
Uses grid search and random search to test different parameter combinations.
"""

import pandas as pd
import numpy as np
import yaml
import json
from itertools import product
from typing import Dict, List, Tuple
import sys
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_backtester import EnhancedBacktester
from utils import filter_trading_hours


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


def run_backtest_with_params(df: pd.DataFrame, base_config: dict, params: dict) -> dict:
    """
    Run backtest with specific parameters.
    
    Args:
        df: DataFrame with OHLCV data
        base_config: Base configuration
        params: Parameters to test
    
    Returns:
        Dictionary with metrics
    """
    # Merge base config with test parameters
    test_config = {**base_config, **params}
    
    try:
        # Suppress print statements during optimization
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            # Initialize backtester
            backtester = EnhancedBacktester(test_config)
            
            # Run backtest
            trades_df = backtester.run_backtest(df)
        
        # Calculate metrics
        if len(trades_df) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'final_equity': test_config['initial_capital'],
                'total_return': 0
            }
        
        metrics = backtester.calculate_metrics(trades_df)
        return metrics
    except Exception as e:
        # Silently skip errors during optimization
        return None


def grid_search_optimization(df: pd.DataFrame, base_config: dict) -> List[Dict]:
    """
    Perform grid search optimization over key parameters.
    
    Args:
        df: DataFrame with OHLCV data
        base_config: Base configuration
    
    Returns:
        List of results sorted by total_pnl
    """
    print("\n" + "="*70)
    print("GRID SEARCH OPTIMIZATION")
    print("="*70 + "\n")
    
    # Define parameter ranges to test
    param_grid = {
        'n_or': [10, 15, 20, 30],  # Opening range period
        'atr_mult': [1.0, 1.5, 2.0, 2.5, 3.0],  # Stop loss multiplier
        'risk_per_trade': [0.002, 0.005, 0.01, 0.02],  # Risk per trade (0.2%, 0.5%, 1%, 2%)
        'rsi_oversold': [20, 30, 40],  # RSI oversold level
        'rsi_overbought': [60, 70, 80],  # RSI overbought level
    }
    
    # Also test minimum score threshold (need to modify strategy for this)
    min_scores = [1, 2, 3, 4]  # Minimum score to enter
    
    results = []
    total_combinations = np.prod([len(v) for v in param_grid.values()]) * len(min_scores)
    current = 0
    
    print(f"Testing {total_combinations} parameter combinations...")
    print("This may take a while...\n")
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for param_combo in product(*param_values):
        params = dict(zip(param_names, param_combo))
        
        for min_score in min_scores:
            current += 1
            if current % 10 == 0:
                print(f"Progress: {current}/{total_combinations} ({100*current/total_combinations:.1f}%)")
            
            # Add min_score to params
            params['min_score'] = min_score
            
            # Run backtest (suppress output)
            metrics = run_backtest_with_params(df, base_config, params)
            
            if metrics is not None:
                result = {
                    **params,
                    **metrics
                }
                results.append(result)
                
                # Print best result so far
                if len(results) == 1 or metrics.get('total_pnl', -float('inf')) > max(r.get('total_pnl', -float('inf')) for r in results[:-1]):
                    print(f"  New best: PnL=₹{metrics.get('total_pnl', 0):.2f}, WinRate={metrics.get('win_rate', 0):.1f}%, Trades={metrics.get('total_trades', 0)}")
    
    # Sort by total_pnl (descending)
    results.sort(key=lambda x: x.get('total_pnl', -float('inf')), reverse=True)
    
    return results


def random_search_optimization(df: pd.DataFrame, base_config: dict, n_iterations: int = 100) -> List[Dict]:
    """
    Perform random search optimization (faster than grid search).
    
    Args:
        df: DataFrame with OHLCV data
        base_config: Base configuration
        n_iterations: Number of random combinations to test
    
    Returns:
        List of results sorted by total_pnl
    """
    print("\n" + "="*70)
    print("RANDOM SEARCH OPTIMIZATION")
    print("="*70 + "\n")
    
    # Define parameter ranges
    param_ranges = {
        'n_or': (10, 30),
        'atr_mult': (1.0, 3.5),
        'risk_per_trade': (0.001, 0.03),
        'rsi_oversold': (20, 45),
        'rsi_overbought': (55, 85),
        'min_score': (1, 5),
    }
    
    results = []
    np.random.seed(42)  # For reproducibility
    
    print(f"Testing {n_iterations} random parameter combinations...\n")
    
    for i in range(n_iterations):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{n_iterations} ({100*(i+1)/n_iterations:.1f}%)")
        
        # Generate random parameters
        params = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            if param_name == 'n_or':
                params[param_name] = int(np.random.uniform(min_val, max_val))
            elif param_name == 'min_score':
                params[param_name] = int(np.random.uniform(min_val, max_val))
            else:
                params[param_name] = np.random.uniform(min_val, max_val)
        
        # Run backtest
        metrics = run_backtest_with_params(df, base_config, params)
        
        if metrics is not None:
            result = {
                **params,
                **metrics
            }
            results.append(result)
    
    # Sort by total_pnl (descending)
    results.sort(key=lambda x: x.get('total_pnl', -float('inf')), reverse=True)
    
    return results


def analyze_results(results: List[Dict], top_n: int = 20) -> pd.DataFrame:
    """
    Analyze and display top results.
    
    Args:
        results: List of result dictionaries
        top_n: Number of top results to display
    
    Returns:
        DataFrame with top results
    """
    if not results:
        print("No results to analyze!")
        return pd.DataFrame()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Display top results
    print("\n" + "="*70)
    print(f"TOP {top_n} PARAMETER COMBINATIONS")
    print("="*70 + "\n")
    
    # Select relevant columns for display
    display_cols = [
        'n_or', 'atr_mult', 'risk_per_trade', 'rsi_oversold', 'rsi_overbought', 'min_score',
        'total_trades', 'win_rate', 'total_pnl', 'profit_factor', 'sharpe_ratio', 'max_drawdown', 'final_equity'
    ]
    
    # Filter to columns that exist
    available_cols = [col for col in display_cols if col in results_df.columns]
    top_results = results_df[available_cols].head(top_n)
    
    # Format for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(top_results.to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70 + "\n")
    
    print(f"Total combinations tested: {len(results)}")
    print(f"Profitable combinations: {len(results_df[results_df['total_pnl'] > 0])}")
    print(f"Best total PnL: ₹{results_df['total_pnl'].max():.2f}")
    print(f"Best win rate: {results_df['win_rate'].max():.2f}%")
    print(f"Best profit factor: {results_df['profit_factor'].max():.2f}")
    print(f"Best Sharpe ratio: {results_df['sharpe_ratio'].max():.2f}")
    
    return results_df


def save_optimization_results(results_df: pd.DataFrame, output_dir: Path):
    """Save optimization results to CSV."""
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f'optimization_results_{timestamp}.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nOptimization results saved to: {output_path}")
    return output_path


def main():
    """Main optimization function."""
    # Get project root directory
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config.yaml'
    data_path = project_root / 'data' / 'nifty_1min.csv'
    
    # Check if files exist
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    # Load configuration
    print("Loading configuration...")
    base_config = load_config(str(config_path))
    
    # Load data
    df = load_data(str(data_path), base_config)
    
    # Ask user for optimization method
    print("\n" + "="*70)
    print("STRATEGY OPTIMIZATION")
    print("="*70)
    print("\nChoose optimization method:")
    print("1. Grid Search (comprehensive but slow)")
    print("2. Random Search (faster, recommended)")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    all_results = []
    
    if choice in ['1', '3']:
        # Grid search
        grid_results = grid_search_optimization(df, base_config)
        all_results.extend(grid_results)
        print(f"\nGrid search completed: {len(grid_results)} combinations tested")
    
    if choice in ['2', '3']:
        # Random search
        n_iterations = input("\nEnter number of random iterations (default 100): ").strip()
        n_iterations = int(n_iterations) if n_iterations else 100
        
        random_results = random_search_optimization(df, base_config, n_iterations)
        all_results.extend(random_results)
        print(f"\nRandom search completed: {n_iterations} combinations tested")
    
    if not all_results:
        print("No results generated!")
        return
    
    # Analyze results
    results_df = analyze_results(all_results, top_n=30)
    
    # Save results
    if len(results_df) > 0:
        output_path = save_optimization_results(results_df, project_root / 'results')
        
        # Save best parameters as new config
        if len(results_df) > 0:
            best_params = results_df.iloc[0]
            best_config = {**base_config}
            
            # Update with best parameters
            for param in ['n_or', 'atr_mult', 'risk_per_trade', 'rsi_oversold', 'rsi_overbought']:
                if param in best_params:
                    best_config[param] = best_params[param]
            
            # Save best config
            best_config_path = project_root / 'config_optimized.yaml'
            with open(best_config_path, 'w') as f:
                yaml.dump(best_config, f, default_flow_style=False)
            print(f"Best parameters saved to: {best_config_path}")
            
            print("\n" + "="*70)
            print("BEST PARAMETERS FOUND")
            print("="*70)
            print(f"Opening Range Period: {best_params.get('n_or', 'N/A')}")
            print(f"ATR Multiplier: {best_params.get('atr_mult', 'N/A')}")
            print(f"Risk Per Trade: {best_params.get('risk_per_trade', 'N/A')}")
            print(f"RSI Oversold: {best_params.get('rsi_oversold', 'N/A')}")
            print(f"RSI Overbought: {best_params.get('rsi_overbought', 'N/A')}")
            print(f"Min Score: {best_params.get('min_score', 'N/A')}")
            print(f"\nExpected Performance:")
            print(f"  Total PnL: ₹{best_params.get('total_pnl', 0):.2f}")
            print(f"  Win Rate: {best_params.get('win_rate', 0):.2f}%")
            print(f"  Profit Factor: {best_params.get('profit_factor', 0):.2f}")
            print(f"  Sharpe Ratio: {best_params.get('sharpe_ratio', 0):.2f}")


if __name__ == '__main__':
    main()

