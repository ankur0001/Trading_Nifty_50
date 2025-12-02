"""
Backtesting engine for the ORB strategy.
Handles trade execution, PnL calculation, and performance metrics.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
import sys
import os

# Handle imports for both module and direct execution
try:
    from .strategy import ORBStrategy
    from .indicators import calculate_vwap, calculate_atr
except ImportError:
    from strategy import ORBStrategy
    from indicators import calculate_vwap, calculate_atr


class Backtester:
    """
    Backtesting engine that simulates trading with realistic execution.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize backtester with configuration.
        
        Args:
            config: Dictionary with backtest parameters
        """
        self.initial_capital = config['initial_capital']
        self.risk_per_trade = config['risk_per_trade']
        self.slippage = config['slippage']
        self.commission = config['commission']
        
        self.strategy = ORBStrategy(config)
        
        # Track state
        self.equity = self.initial_capital
        self.cash = self.initial_capital
        self.position = 0  # Number of shares/units
        self.position_side = None  # 'LONG' or 'SHORT'
        self.entry_price = None  # Store entry price in backtester
        self.entry_time = None  # Store entry time in backtester
        
        # Results
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """
        Calculate position size based on risk per trade.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
        
        Returns:
            Number of units to trade
        """
        risk_amount = self.equity * self.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
        
        position_size = int(risk_amount / risk_per_unit)
        return max(1, position_size)  # At least 1 unit
    
    def apply_slippage(self, price: float, side: str) -> float:
        """
        Apply slippage to execution price.
        
        Args:
            price: Base price
            side: 'LONG' (buy) or 'SHORT' (sell)
        
        Returns:
            Price with slippage applied
        """
        if side == 'LONG':
            return price * (1 + self.slippage)
        else:  # SHORT
            return price * (1 - self.slippage)
    
    def calculate_commission(self, price: float, quantity: int) -> float:
        """
        Calculate commission cost.
        
        Args:
            price: Execution price
            quantity: Number of units
        
        Returns:
            Commission amount
        """
        return price * quantity * self.commission
    
    def execute_entry(self, side: str, entry_price: float, entry_time: pd.Timestamp, 
                     atr: float) -> Dict:
        """
        Execute a trade entry.
        
        Args:
            side: 'LONG' or 'SHORT'
            entry_price: Base entry price (next bar's open)
            entry_time: Entry timestamp
            atr: Current ATR value
        
        Returns:
            Dictionary with entry details
        """
        # Apply slippage to get actual fill price
        fill_price = self.apply_slippage(entry_price, side)
        
        # Calculate stop loss based on fill price (more realistic)
        stop_loss = self.strategy.calculate_stop_loss(fill_price, side, atr)
        
        # Calculate position size based on fill price and stop loss
        quantity = self.calculate_position_size(fill_price, stop_loss)
        
        # Calculate commission
        commission_cost = self.calculate_commission(fill_price, quantity)
        
        # Update position
        if side == 'LONG':
            cost = fill_price * quantity + commission_cost
            if cost <= self.cash:
                self.position = quantity
                self.position_side = 'LONG'
                self.cash -= cost
            else:
                # Not enough cash, reduce position size
                quantity = int(self.cash / (fill_price * (1 + self.commission)))
                if quantity > 0:
                    cost = fill_price * quantity + self.calculate_commission(fill_price, quantity)
                    self.position = quantity
                    self.position_side = 'LONG'
                    self.cash -= cost
                else:
                    return None
        else:  # SHORT
            # For short, we receive cash (simplified - no margin requirements)
            proceeds = fill_price * quantity - commission_cost
            self.position = quantity
            self.position_side = 'SHORT'
            self.cash += proceeds
        
        # Store entry details in backtester
        self.entry_price = fill_price
        self.entry_time = entry_time
        
        # Update strategy state
        self.strategy.enter_position(side, fill_price, entry_time, atr)
        
        return {
            'side': side,
            'quantity': quantity,
            'entry_price': fill_price,
            'entry_time': entry_time,
            'stop_loss': stop_loss,
            'commission': commission_cost
        }
    
    def execute_exit(self, exit_price: float, exit_time: pd.Timestamp, 
                    reason: str) -> Dict:
        """
        Execute a trade exit.
        
        Args:
            exit_price: Exit price
            exit_time: Exit timestamp
            reason: Reason for exit
        
        Returns:
            Dictionary with exit details
        """
        if self.position == 0:
            return None
        
        side = self.position_side
        quantity = self.position
        entry_price = self.entry_price
        entry_time = self.entry_time
        
        # Safety check
        if entry_price is None:
            # Fallback to strategy if backtester doesn't have it
            entry_price = self.strategy.entry_price
            entry_time = self.strategy.entry_time
        
        if entry_price is None:
            # Still None, can't calculate PnL
            return None
        
        # Apply slippage
        fill_price = self.apply_slippage(exit_price, 'SHORT' if side == 'LONG' else 'LONG')
        
        # Calculate commission
        commission_cost = self.calculate_commission(fill_price, quantity)
        
        # Calculate PnL
        if side == 'LONG':
            proceeds = fill_price * quantity - commission_cost
            pnl = proceeds - (entry_price * quantity)
        else:  # SHORT
            cost = fill_price * quantity + commission_cost
            pnl = (entry_price * quantity) - cost
        
        # Update cash and position
        if side == 'LONG':
            self.cash += proceeds
        else:  # SHORT
            self.cash -= cost
        
        self.position = 0
        self.position_side = None
        self.entry_price = None
        self.entry_time = None
        
        # Exit strategy position
        self.strategy.exit_position()
        
        # Calculate total commission (entry + exit)
        total_commission = self.calculate_commission(entry_price, quantity) + commission_cost
        
        return {
            'exit_price': fill_price,
            'exit_time': exit_time,
            'pnl': pnl,
            'exit_reason': reason,
            'total_commission': total_commission
        }
    
    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the backtest on the provided data.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with trade results
        """
        # Prepare data
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate indicators
        df['vwap'] = calculate_vwap(df)
        df['atr'] = calculate_atr(df, self.strategy.atr_period)
        
        # Add date column for grouping
        df['date_only'] = df['date'].dt.date
        
        # Track current trade
        current_trade = None
        
        # Iterate through data
        for i in range(len(df)):
            row = df.iloc[i]
            current_time = row['date']
            current_date = current_time.date()
            
            # Reset daily state
            self.strategy.reset_daily_state(current_date)
            
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
                        # Last bar, use close
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
                    # Check entry conditions
                    entry_signal = self.strategy.check_entry_conditions(
                        row, or_high, or_low, row['vwap']
                    )
                    
                    if entry_signal:
                        # Execute entry (fill at next bar's open)
                        if i + 1 < len(df):
                            next_bar = df.iloc[i + 1]
                            entry_result = self.execute_entry(
                                entry_signal, next_bar['open'], next_bar['date'], row['atr']
                            )
                        else:
                            # Last bar, use close
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
    
    def calculate_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            trades_df: DataFrame with trade results
        
        Returns:
            Dictionary with performance metrics
        """
        if len(trades_df) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'final_equity': self.initial_capital,
                'total_return': 0.0
            }
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
        gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum()) if losing_trades > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Equity curve metrics
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df = equity_df.set_index('date')
        
        # Calculate drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = equity_df['equity'] - equity_df['cummax']
        equity_df['drawdown_pct'] = (equity_df['drawdown'] / equity_df['cummax']) * 100
        max_drawdown = equity_df['drawdown_pct'].min()
        
        # Calculate daily returns for Sharpe ratio
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        daily_returns = equity_df['daily_return'].dropna()
        
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Final equity and return
        final_equity = equity_df['equity'].iloc[-1]
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_equity': final_equity,
            'total_return': total_return,
            'initial_capital': self.initial_capital
        }

