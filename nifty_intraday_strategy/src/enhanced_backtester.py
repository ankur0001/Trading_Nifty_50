"""
Enhanced backtesting engine that uses the enhanced strategy with all indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys
import os

# Handle imports for both module and direct execution
try:
    from .backtester import Backtester
    from .enhanced_strategy import EnhancedStrategy
    from .indicators import calculate_vwap, calculate_atr
except ImportError:
    from backtester import Backtester
    from enhanced_strategy import EnhancedStrategy
    from indicators import calculate_vwap, calculate_atr


class EnhancedBacktester(Backtester):
    """
    Enhanced backtester that uses EnhancedStrategy with all indicators.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize enhanced backtester.
        
        Args:
            config: Dictionary with backtest parameters
        """
        # Initialize with enhanced strategy
        self.initial_capital = config['initial_capital']
        self.risk_per_trade = config['risk_per_trade']
        self.slippage = config['slippage']
        self.commission = config['commission']
        
        self.strategy = EnhancedStrategy(config)
        
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
    
    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the enhanced backtest.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with trade results
        """
        # Prepare data
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate base indicators
        print("Calculating base indicators (VWAP, ATR)...")
        df['vwap'] = calculate_vwap(df)
        df['atr'] = calculate_atr(df, self.strategy.atr_period)
        
        # Calculate all enhanced indicators
        print("Calculating enhanced indicators and patterns...")
        self.strategy.calculate_all_indicators(df)
        
        # Add date column for grouping
        df['date_only'] = df['date'].dt.date
        
        # Track current trade
        current_trade = None
        
        # Iterate through data
        print("Running backtest...")
        total_bars = len(df)
        prev_date = None
        
        for i in range(len(df)):
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1}/{total_bars} bars ({100*(i+1)/total_bars:.1f}%)")
            
            row = df.iloc[i]
            current_time = row['date']
            current_date = current_time.date()
            
            # Check if date changed - close any open position from previous day
            if current_trade is not None and prev_date is not None and current_date != prev_date:
                # Date changed, force exit any open position at previous day's close
                prev_row = df.iloc[i-1]
                exit_result = self.execute_exit(prev_row['close'], prev_row['date'], 'End of Day')
                if exit_result:
                    trade = {
                        **current_trade,
                        **exit_result
                    }
                    trade['pnl'] = exit_result['pnl']
                    trade['return_pct'] = (exit_result['pnl'] / (current_trade['entry_price'] * current_trade['quantity'])) * 100
                    self.trades.append(trade)
                    current_trade = None
                    # Update equity
                    self.equity = self.cash + (self.position * prev_row['close'] if self.position > 0 else 0)
            
            # Reset daily state
            self.strategy.reset_daily_state(current_date)
            
            # Update prev_date for next iteration
            prev_date = current_date
            
            # Skip if missing indicators
            if pd.isna(row['vwap']) or pd.isna(row['atr']):
                self.equity_curve.append({
                    'date': current_time,
                    'equity': self.equity,
                    'cash': self.cash,
                    'position_value': self.position * row['close'] if self.position > 0 else 0
                })
                continue
            
            # Check for exit first (if in position)
            if current_trade is not None:
                should_exit = self.strategy.check_exit_conditions(
                    row, row['vwap'], current_time
                )
                
                if should_exit:
                    # Determine exit price and reason
                    if self.position_side == 'LONG':
                        if row['low'] <= self.strategy.stop_loss:
                            exit_price = self.strategy.stop_loss
                            reason = 'Stop Loss'
                        elif row['close'] < row['vwap']:
                            exit_price = row['close']
                            reason = 'VWAP Cross'
                        else:  # End of day
                            exit_price = row['close']
                            reason = 'End of Day'
                    else:  # SHORT
                        if row['high'] >= self.strategy.stop_loss:
                            exit_price = self.strategy.stop_loss
                            reason = 'Stop Loss'
                        elif row['close'] > row['vwap']:
                            exit_price = row['close']
                            reason = 'VWAP Cross'
                        else:  # End of day
                            exit_price = row['close']
                            reason = 'End of Day'
                    
                    # Execute exit (fill at next bar's open)
                    if i + 1 < len(df):
                        next_bar = df.iloc[i + 1]
                        exit_result = self.execute_exit(next_bar['open'], next_bar['date'], reason)
                    else:
                        exit_result = self.execute_exit(row['close'], current_time, reason)
                    
                    if exit_result:
                        # Complete trade
                        trade = {
                            **current_trade,
                            **exit_result
                        }
                        trade['pnl'] = exit_result['pnl']
                        trade['return_pct'] = (exit_result['pnl'] / (current_trade['entry_price'] * current_trade['quantity'])) * 100
                        self.trades.append(trade)
                        
                        # Update equity
                        self.equity = self.cash + (self.position * row['close'] if self.position > 0 else 0)
                        
                        current_trade = None
            
            # Check for entry (if no position)
            if current_trade is None and self.position == 0:
                # Calculate opening range for this day
                or_high, or_low = self.strategy.calculate_opening_range(df, current_time)
                
                if or_high is not None and or_low is not None:
                    # Check enhanced entry conditions
                    entry_signal = self.strategy.check_enhanced_entry_conditions(
                        row, or_high, or_low, row['vwap'], i
                    )
                    
                    if entry_signal:
                        # Execute entry (fill at next bar's open)
                        if i + 1 < len(df):
                            next_bar = df.iloc[i + 1]
                            entry_result = self.execute_entry(
                                entry_signal, next_bar['open'], next_bar['date'], row['atr']
                            )
                        else:
                            entry_result = self.execute_entry(
                                entry_signal, row['close'], current_time, row['atr']
                            )
                        
                        if entry_result:
                            current_trade = entry_result
                            # Update equity
                            self.equity = self.cash + (self.position * row['close'] if self.position > 0 else 0)
            
            # Update equity curve
            position_value = self.position * row['close'] if self.position > 0 else 0
            self.equity = self.cash + position_value
            
            self.equity_curve.append({
                'date': current_time,
                'equity': self.equity,
                'cash': self.cash,
                'position_value': position_value
            })
        
        # Close any open position at the end
        if current_trade is not None:
            last_row = df.iloc[-1]
            exit_result = self.execute_exit(last_row['close'], last_row['date'], 'End of Data')
            if exit_result:
                trade = {
                    **current_trade,
                    **exit_result
                }
                trade['pnl'] = exit_result['pnl']
                trade['return_pct'] = (exit_result['pnl'] / (current_trade['entry_price'] * current_trade['quantity'])) * 100
                self.trades.append(trade)
        
        # Convert trades to DataFrame
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
        else:
            trades_df = pd.DataFrame()
        
        return trades_df

