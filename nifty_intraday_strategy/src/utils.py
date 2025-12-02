"""
Utility functions for date filtering and plotting.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def filter_trading_hours(df: pd.DataFrame, start_time: str = "09:15", 
                        end_time: str = "15:15") -> pd.DataFrame:
    """
    Filter DataFrame to include only trading hours.
    
    Args:
        df: DataFrame with 'date' column
        start_time: Start time in HH:MM format
        end_time: End time in HH:MM format
    
    Returns:
        Filtered DataFrame
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['time'] = df['date'].dt.time
    
    start = pd.to_datetime(start_time).time()
    end = pd.to_datetime(end_time).time()
    
    mask = (df['time'] >= start) & (df['time'] <= end)
    return df[mask].reset_index(drop=True)


def plot_equity_curve(equity_curve: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot equity curve.
    
    Args:
        equity_curve: DataFrame with 'date' and 'equity' columns
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(equity_curve['date'], equity_curve['equity'], linewidth=1.5, color='#2E86AB')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Equity (₹)', fontsize=12)
    ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_pnl_histogram(trades_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot PnL histogram.
    
    Args:
        trades_df: DataFrame with 'pnl' column
        save_path: Optional path to save the plot
    """
    if len(trades_df) == 0:
        print("No trades to plot.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pnl_values = trades_df['pnl'].values
    
    # Create histogram with separate colors for positive and negative
    # Use numpy to separate positive and negative values
    positive_pnl = pnl_values[pnl_values >= 0]
    negative_pnl = pnl_values[pnl_values < 0]
    
    # Determine appropriate number of bins
    n_bins = min(50, max(10, len(pnl_values) // 5))
    
    # Plot positive and negative separately
    if len(positive_pnl) > 0:
        ax.hist(positive_pnl, bins=n_bins, color='#06A77D', alpha=0.7, edgecolor='black', linewidth=0.5, label='Profit')
    if len(negative_pnl) > 0:
        ax.hist(negative_pnl, bins=n_bins, color='#D00000', alpha=0.7, edgecolor='black', linewidth=0.5, label='Loss')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('PnL (₹)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('PnL Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_pnl = pnl_values.mean()
    median_pnl = np.median(pnl_values)
    ax.axvline(x=mean_pnl, color='blue', linestyle='--', linewidth=1.5, label=f'Mean: ₹{mean_pnl:.2f}')
    ax.axvline(x=median_pnl, color='orange', linestyle='--', linewidth=1.5, label=f'Median: ₹{median_pnl:.2f}')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_backtest_summary(metrics: dict, trades_df: pd.DataFrame):
    """
    Print formatted backtest summary.
    
    Args:
        metrics: Dictionary with performance metrics
        trades_df: DataFrame with trade results
    """
    print("\n" + "="*70)
    print("BACKTEST SUMMARY")
    print("="*70)
    
    print(f"\n📊 CAPITAL & RETURNS")
    print(f"  Initial Capital:     ₹{metrics['initial_capital']:,.2f}")
    print(f"  Final Equity:        ₹{metrics['final_equity']:,.2f}")
    print(f"  Total Return:        {metrics['total_return']:.2f}%")
    print(f"  Total PnL:           ₹{metrics['total_pnl']:,.2f}")
    
    print(f"\n📈 TRADE STATISTICS")
    print(f"  Total Trades:        {metrics['total_trades']}")
    print(f"  Winning Trades:      {metrics['winning_trades']}")
    print(f"  Losing Trades:       {metrics['losing_trades']}")
    print(f"  Win Rate:            {metrics['win_rate']:.2f}%")
    
    print(f"\n💰 PnL METRICS")
    print(f"  Average PnL/Trade:   ₹{metrics['avg_pnl']:,.2f}")
    print(f"  Average Win:         ₹{metrics['avg_win']:,.2f}")
    print(f"  Average Loss:        ₹{metrics['avg_loss']:,.2f}")
    print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")
    
    print(f"\n📉 RISK METRICS")
    print(f"  Max Drawdown:        {metrics['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
    
    if len(trades_df) > 0:
        print(f"\n📋 TRADE LIST (First 10 trades)")
        print("-"*70)
        display_cols = ['entry_time', 'exit_time', 'side', 'quantity', 
                       'entry_price', 'exit_price', 'pnl', 'return_pct', 'exit_reason']
        available_cols = [col for col in display_cols if col in trades_df.columns]
        print(trades_df[available_cols].head(10).to_string(index=False))
        
        if len(trades_df) > 10:
            print(f"\n... and {len(trades_df) - 10} more trades")
    
    print("\n" + "="*70 + "\n")

