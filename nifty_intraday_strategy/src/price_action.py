"""
Price action tools module.
Implements Pivot Points and Fibonacci Retracement levels.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional


def calculate_pivot_points(df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """
    Calculate Pivot Points (Standard, Fibonacci, or Camarilla).
    
    Args:
        df: DataFrame with OHLC data
        method: 'standard', 'fibonacci', or 'camarilla'
    
    Returns:
        DataFrame with pivot point levels
    """
    result = pd.DataFrame(index=df.index)
    
    # Calculate previous period's high, low, close
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    prev_close = df['close'].shift(1)
    
    if method == 'standard':
        # Standard Pivot Points
        result['pivot'] = (prev_high + prev_low + prev_close) / 3
        result['r1'] = 2 * result['pivot'] - prev_low
        result['r2'] = result['pivot'] + (prev_high - prev_low)
        result['r3'] = prev_high + 2 * (result['pivot'] - prev_low)
        result['s1'] = 2 * result['pivot'] - prev_high
        result['s2'] = result['pivot'] - (prev_high - prev_low)
        result['s3'] = prev_low - 2 * (prev_high - result['pivot'])
    
    elif method == 'fibonacci':
        # Fibonacci Pivot Points
        result['pivot'] = (prev_high + prev_low + prev_close) / 3
        diff = prev_high - prev_low
        result['r1'] = result['pivot'] + 0.382 * diff
        result['r2'] = result['pivot'] + 0.618 * diff
        result['r3'] = result['pivot'] + 1.000 * diff
        result['s1'] = result['pivot'] - 0.382 * diff
        result['s2'] = result['pivot'] - 0.618 * diff
        result['s3'] = result['pivot'] - 1.000 * diff
    
    elif method == 'camarilla':
        # Camarilla Pivot Points
        result['pivot'] = (prev_high + prev_low + prev_close) / 3
        diff = prev_high - prev_low
        result['r1'] = prev_close + diff * 1.1 / 12
        result['r2'] = prev_close + diff * 1.1 / 6
        result['r3'] = prev_close + diff * 1.1 / 4
        result['r4'] = prev_close + diff * 1.1 / 2
        result['s1'] = prev_close - diff * 1.1 / 12
        result['s2'] = prev_close - diff * 1.1 / 6
        result['s3'] = prev_close - diff * 1.1 / 4
        result['s4'] = prev_close - diff * 1.1 / 2
    
    return result


def calculate_fibonacci_retracement(high: float, low: float, 
                                    trend: str = 'up') -> Dict[str, float]:
    """
    Calculate Fibonacci Retracement levels.
    
    Args:
        high: High price
        low: Low price
        trend: 'up' for uptrend, 'down' for downtrend
    
    Returns:
        Dictionary with Fibonacci levels
    """
    diff = high - low
    fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    if trend == 'up':
        # Retracement from high to low
        levels = {f'fib_{int(l*1000)}': high - (diff * l) for l in fib_levels}
    else:
        # Retracement from low to high
        levels = {f'fib_{int(l*1000)}': low + (diff * l) for l in fib_levels}
    
    return levels


def calculate_fibonacci_retracement_series(df: pd.DataFrame, period: int = 20, 
                                          trend: str = 'auto') -> pd.DataFrame:
    """
    Calculate Fibonacci Retracement levels for a series.
    
    Args:
        df: DataFrame with OHLC data
        period: Period to look back for high/low
        trend: 'up', 'down', or 'auto'
    
    Returns:
        DataFrame with Fibonacci levels
    """
    result = pd.DataFrame(index=df.index)
    
    # Calculate rolling high and low
    rolling_high = df['high'].rolling(window=period, min_periods=1).max()
    rolling_low = df['low'].rolling(window=period, min_periods=1).min()
    
    # Determine trend
    if trend == 'auto':
        trend_direction = (df['close'] > df['close'].shift(period)).map({True: 'up', False: 'down'})
    else:
        trend_direction = pd.Series(trend, index=df.index)
    
    # Calculate Fibonacci levels
    fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    for level in fib_levels:
        col_name = f'fib_{int(level*1000)}'
        
        # For uptrend: retracement from high
        up_mask = trend_direction == 'up'
        result.loc[up_mask, col_name] = rolling_high[up_mask] - \
                                        ((rolling_high[up_mask] - rolling_low[up_mask]) * level)
        
        # For downtrend: retracement from low
        down_mask = trend_direction == 'down'
        result.loc[down_mask, col_name] = rolling_low[down_mask] + \
                                         ((rolling_high[down_mask] - rolling_low[down_mask]) * level)
    
    # Calculate which Fibonacci level current price is near
    result['fib_level'] = 0.0
    for i, level in enumerate(fib_levels):
        if i < len(fib_levels) - 1:
            next_level = fib_levels[i + 1]
            col_name = f'fib_{int(level*1000)}'
            next_col_name = f'fib_{int(next_level*1000)}'
            
            mask = (df['close'] >= result[col_name]) & (df['close'] < result[next_col_name])
            result.loc[mask, 'fib_level'] = level
    
    return result


def identify_support_resistance(df: pd.DataFrame, window: int = 20, 
                               tolerance: float = 0.01) -> pd.DataFrame:
    """
    Identify support and resistance levels using pivot points and price clustering.
    
    Args:
        df: DataFrame with OHLC data
        window: Window for identifying levels
        tolerance: Price tolerance for clustering
    
    Returns:
        DataFrame with support and resistance levels
    """
    result = pd.DataFrame(index=df.index)
    
    # Use pivot points as base
    pivots = calculate_pivot_points(df, method='standard')
    
    # Identify support (price bounces up from level)
    result['support_level'] = pivots[['s1', 's2', 's3']].min(axis=1)
    result['resistance_level'] = pivots[['r1', 'r2', 'r3']].max(axis=1)
    
    # Calculate distance to nearest support/resistance
    result['dist_to_support'] = (df['close'] - result['support_level']) / df['close']
    result['dist_to_resistance'] = (result['resistance_level'] - df['close']) / df['close']
    
    # Identify if price is near support or resistance
    result['near_support'] = (result['dist_to_support'].abs() < tolerance).astype(int)
    result['near_resistance'] = (result['dist_to_resistance'].abs() < tolerance).astype(int)
    
    return result


def get_all_price_action(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Calculate all price action tools.
    
    Args:
        df: DataFrame with OHLC data
        config: Optional configuration dict
    
    Returns:
        DataFrame with all price action indicators
    """
    if config is None:
        config = {}
    
    price_action = pd.DataFrame(index=df.index)
    
    # Pivot Points
    pivots_std = calculate_pivot_points(df, method='standard')
    pivots_fib = calculate_pivot_points(df, method='fibonacci')
    pivots_cam = calculate_pivot_points(df, method='camarilla')
    
    price_action = pd.concat([price_action, pivots_std.add_suffix('_std')], axis=1)
    price_action = pd.concat([price_action, pivots_fib.add_suffix('_fib')], axis=1)
    price_action = pd.concat([price_action, pivots_cam.add_suffix('_cam')], axis=1)
    
    # Fibonacci Retracement
    fib_period = config.get('fib_period', 20)
    fib_ret = calculate_fibonacci_retracement_series(df, period=fib_period)
    price_action = pd.concat([price_action, fib_ret], axis=1)
    
    # Support/Resistance
    sr = identify_support_resistance(df, 
                                    window=config.get('sr_window', 20),
                                    tolerance=config.get('sr_tolerance', 0.01))
    price_action = pd.concat([price_action, sr], axis=1)
    
    return price_action

