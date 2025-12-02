"""
Technical indicators for the backtesting strategy.
Implements VWAP and ATR calculations.
"""

import pandas as pd
import numpy as np


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP) for intraday data.
    VWAP resets each day at market open.
    
    Args:
        df: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
    
    Returns:
        Series with VWAP values
    """
    df = df.copy()
    df['date_only'] = pd.to_datetime(df['date']).dt.date
    
    # Calculate typical price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate cumulative volume and price*volume per day
    df['pv'] = df['typical_price'] * df['volume']
    
    # Group by date and calculate cumulative sums
    df['cum_pv'] = df.groupby('date_only')['pv'].cumsum()
    df['cum_volume'] = df.groupby('date_only')['volume'].cumsum()
    
    # Calculate VWAP (avoid division by zero)
    df['vwap'] = np.where(df['cum_volume'] > 0, 
                          df['cum_pv'] / df['cum_volume'], 
                          df['typical_price'])
    
    return df['vwap']


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) indicator.
    
    Args:
        df: DataFrame with columns ['date', 'open', 'high', 'low', 'close']
        period: Period for ATR calculation (default: 14)
    
    Returns:
        Series with ATR values
    """
    df = df.copy()
    
    # Calculate True Range components
    df['prev_close'] = df['close'].shift(1)
    
    # True Range = max of:
    # 1. High - Low
    # 2. abs(High - Previous Close)
    # 3. abs(Low - Previous Close)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate ATR as rolling mean of True Range
    df['atr'] = df['true_range'].rolling(window=period, min_periods=1).mean()
    
    return df['atr']

