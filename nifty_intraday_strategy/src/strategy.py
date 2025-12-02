"""
Opening Range Breakout (ORB) strategy with VWAP and ATR filters.
Implements the trading logic for entry, stop loss, and exit conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import time


class ORBStrategy:
    """
    Opening Range Breakout strategy with VWAP and ATR filters.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize strategy with configuration parameters.
        
        Args:
            config: Dictionary with strategy parameters
        """
        self.n_or = config['n_or']  # Opening range period in minutes
        self.atr_period = config['atr_period']
        self.atr_mult = config['atr_mult']
        self.start_time = pd.to_datetime(config['start_time']).time()
        self.end_time = pd.to_datetime(config['end_time']).time()
        self.max_trades_per_day = config.get('max_trades_per_day', 1)
        
        # Track state
        self.current_position = None  # None, 'LONG', or 'SHORT'
        self.entry_price = None
        self.entry_time = None
        self.stop_loss = None
        self.trades_today = 0
        self.current_date = None
        
    def reset_daily_state(self, current_date):
        """Reset daily state variables."""
        if self.current_date != current_date:
            self.current_date = current_date
            self.trades_today = 0
            # Close any open position at end of day
            if self.current_position is not None:
                self.current_position = None
                self.entry_price = None
                self.entry_time = None
                self.stop_loss = None
    
    def calculate_opening_range(self, df: pd.DataFrame, date: pd.Timestamp) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate opening range high and low for a given date.
        
        Args:
            df: DataFrame with OHLCV data
            date: Date to calculate OR for
        
        Returns:
            Tuple of (OR_high, OR_low) or (None, None) if insufficient data
        """
        date_str = date.date()
        df['date_only'] = pd.to_datetime(df['date']).dt.date
        
        # Filter data for this date
        day_data = df[df['date_only'] == date_str].copy()
        
        if len(day_data) < self.n_or:
            return None, None
        
        # Get first n_or minutes after start_time
        day_data = day_data.sort_values('date')
        or_data = day_data.head(self.n_or)
        
        or_high = or_data['high'].max()
        or_low = or_data['low'].min()
        
        return or_high, or_low
    
    def check_entry_conditions(self, row: pd.Series, or_high: float, or_low: float, 
                               vwap: float) -> Optional[str]:
        """
        Check if entry conditions are met.
        
        Args:
            row: Current bar data
            or_high: Opening range high
            or_low: Opening range low
            vwap: Current VWAP value
        
        Returns:
            'LONG', 'SHORT', or None
        """
        if self.trades_today >= self.max_trades_per_day:
            return None
        
        if self.current_position is not None:
            return None
        
        close = row['close']
        
        # LONG entry: close above OR high AND close > VWAP
        if close > or_high and close > vwap:
            return 'LONG'
        
        # SHORT entry: close below OR low AND close < VWAP
        if close < or_low and close < vwap:
            return 'SHORT'
        
        return None
    
    def calculate_stop_loss(self, entry_price: float, side: str, atr: float) -> float:
        """
        Calculate stop loss price based on ATR.
        
        Args:
            entry_price: Entry price
            side: 'LONG' or 'SHORT'
            atr: Current ATR value
        
        Returns:
            Stop loss price
        """
        if side == 'LONG':
            return entry_price - (self.atr_mult * atr)
        else:  # SHORT
            return entry_price + (self.atr_mult * atr)
    
    def check_exit_conditions(self, row: pd.Series, vwap: float, 
                             current_time: pd.Timestamp) -> bool:
        """
        Check if exit conditions are met.
        
        Args:
            row: Current bar data
            vwap: Current VWAP value
            current_time: Current timestamp
        
        Returns:
            True if should exit, False otherwise
        """
        if self.current_position is None:
            return False
        
        # Check stop loss
        if self.current_position == 'LONG':
            if row['low'] <= self.stop_loss:
                return True
            # VWAP cross back (close below VWAP)
            if row['close'] < vwap:
                return True
        else:  # SHORT
            if row['high'] >= self.stop_loss:
                return True
            # VWAP cross back (close above VWAP)
            if row['close'] > vwap:
                return True
        
        # End of day exit
        if current_time.time() >= self.end_time:
            return True
        
        return False
    
    def enter_position(self, side: str, entry_price: float, entry_time: pd.Timestamp, 
                      atr: float):
        """
        Enter a new position.
        
        Args:
            side: 'LONG' or 'SHORT'
            entry_price: Entry price
            entry_time: Entry timestamp
            atr: Current ATR value
        """
        self.current_position = side
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = self.calculate_stop_loss(entry_price, side, atr)
        self.trades_today += 1
    
    def exit_position(self):
        """Exit current position."""
        side = self.current_position
        entry_price = self.entry_price
        entry_time = self.entry_time
        
        self.current_position = None
        self.entry_price = None
        self.entry_time = None
        self.stop_loss = None
        
        return side, entry_price, entry_time

