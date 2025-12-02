"""
Main script to run the Nifty intraday backtest.
"""

import pandas as pd
import yaml
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtester import Backtester
from utils import filter_trading_hours, plot_equity_curve, plot_pnl_histogram, print_backtest_summary


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
    
    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file.
    
    Args:
        data_path: Path to CSV file
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter trading hours
    config = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml'))
    df = filter_trading_hours(df, config['start_time'], config['end_time'])
    
    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def main():
    """Main function to run the backtest."""
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
        print(f"Please place your NIFTY 50_minute.csv file in the data/ directory and rename it to nifty_1min.csv")
        return
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(str(config_path))
    
    # Load data
    df = load_data(str(data_path))
    
    # Initialize backtester
    print("\nInitializing backtester...")
    backtester = Backtester(config)
    
    # Run backtest
    print("Running backtest...")
    print("This may take a few minutes for large datasets...\n")
    trades_df = backtester.run_backtest(df)
    
    # Calculate metrics
    print("Calculating performance metrics...")
    metrics = backtester.calculate_metrics(trades_df)
    
    # Print summary
    print_backtest_summary(metrics, trades_df)
    
    # Generate plots (if enabled)
    generate_plots = config.get('generate_plots', True)
    save_results = config.get('save_results', True)
    
    if generate_plots:
        print("Generating plots...")
        equity_curve_df = pd.DataFrame(backtester.equity_curve)
        
        if len(equity_curve_df) > 0:
            plot_equity_curve(equity_curve_df)
        
        if len(trades_df) > 0:
            plot_pnl_histogram(trades_df)
    else:
        print("Skipping plots (disabled in config)")
    
    # Save results (if enabled)
    if save_results:
        results_dir = project_root / 'results'
        results_dir.mkdir(exist_ok=True)
        
        equity_curve_df = pd.DataFrame(backtester.equity_curve)
        
        if len(trades_df) > 0:
            trades_df.to_csv(results_dir / 'trades.csv', index=False)
            print(f"\nTrades saved to {results_dir / 'trades.csv'}")
        
        equity_curve_df.to_csv(results_dir / 'equity_curve.csv', index=False)
        print(f"Equity curve saved to {results_dir / 'equity_curve.csv'}")
        
        # Save metrics
        import json
        with open(results_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {results_dir / 'metrics.json'}")
    else:
        print("Skipping file saves (disabled in config)")
    
    print("\n✅ Backtest completed successfully!")


if __name__ == '__main__':
    main()

