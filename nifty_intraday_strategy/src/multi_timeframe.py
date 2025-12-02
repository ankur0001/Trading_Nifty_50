"""
Multiple timeframe analysis module.
Aggregates data to different timeframes and analyzes across them.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import sys
import os

# Handle imports for both module and direct execution
def _import_price_action():
    try:
        from .price_action import calculate_pivot_points
        return calculate_pivot_points
    except ImportError:
        from price_action import calculate_pivot_points
        return calculate_pivot_points

def _import_indicators():
    try:
        from .advanced_indicators import calculate_rsi, calculate_macd
        return calculate_rsi, calculate_macd
    except ImportError:
        from advanced_indicators import calculate_rsi, calculate_macd
        return calculate_rsi, calculate_macd


def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample 1-minute data to different timeframes.
    
    Args:
        df: DataFrame with 1-minute OHLCV data
        timeframe: Target timeframe ('15min', '1H', '1D', '1W')
    
    Returns:
        Resampled DataFrame
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # Map timeframe strings to pandas offset aliases
    timeframe_map = {
        '15min': '15T',
        '1H': '1H',
        '1D': '1D',
        '1W': '1W'
    }
    
    if timeframe not in timeframe_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    offset = timeframe_map[timeframe]
    
    # Resample OHLCV data
    resampled = pd.DataFrame()
    resampled['open'] = df['open'].resample(offset).first()
    resampled['high'] = df['high'].resample(offset).max()
    resampled['low'] = df['low'].resample(offset).min()
    resampled['close'] = df['close'].resample(offset).last()
    resampled['volume'] = df['volume'].resample(offset).sum()
    
    # Reset index to have 'date' column
    resampled = resampled.reset_index()
    resampled.rename(columns={'date': 'date'}, inplace=True)
    
    return resampled


def get_trend_multiple_timeframes(df: pd.DataFrame, 
                                 timeframes: List[str] = ['15min', '1H', '1D']) -> pd.DataFrame:
    """
    Analyze trend across multiple timeframes.
    
    Args:
        df: DataFrame with 1-minute OHLCV data
        timeframes: List of timeframes to analyze
    
    Returns:
        DataFrame with trend signals from each timeframe
    """
    result = pd.DataFrame(index=df.index)
    
    original_date_col = 'date' if 'date' in df.columns else df.index.name
    
    for tf in timeframes:
        try:
            # Resample to timeframe
            resampled = resample_data(df, tf)
            
            # Calculate moving averages for trend
            resampled['sma_20'] = resampled['close'].rolling(window=20, min_periods=1).mean()
            resampled['sma_50'] = resampled['close'].rolling(window=50, min_periods=1).mean()
            
            # Determine trend
            resampled['trend'] = 0
            resampled.loc[resampled['sma_20'] > resampled['sma_50'], 'trend'] = 1  # Uptrend
            resampled.loc[resampled['sma_20'] < resampled['sma_50'], 'trend'] = -1  # Downtrend
            
            # Map back to 1-minute data
            resampled = resampled.set_index('date')
            resampled = resampled.reindex(df.index, method='ffill')
            
            result[f'trend_{tf}'] = resampled['trend']
            result[f'sma20_{tf}'] = resampled['sma_20']
            result[f'sma50_{tf}'] = resampled['sma_50']
            
        except Exception as e:
            print(f"Warning: Could not process timeframe {tf}: {e}")
            result[f'trend_{tf}'] = 0
            result[f'sma20_{tf}'] = df['close']
            result[f'sma50_{tf}'] = df['close']
    
    # Combined trend signal (all timeframes aligned)
    bullish_tfs = [f'trend_{tf}' for tf in timeframes if f'trend_{tf}' in result.columns]
    if bullish_tfs:
        result['trend_aligned'] = (result[bullish_tfs].sum(axis=1) > 0).astype(int) - \
                                  (result[bullish_tfs].sum(axis=1) < 0).astype(int)
    
    return result


def get_support_resistance_multiple_timeframes(df: pd.DataFrame,
                                              timeframes: List[str] = ['15min', '1H', '1D']) -> pd.DataFrame:
    """
    Identify support and resistance levels across multiple timeframes.
    
    Args:
        df: DataFrame with 1-minute OHLCV data
        timeframes: List of timeframes to analyze
    
    Returns:
        DataFrame with support/resistance levels
    """
    result = pd.DataFrame(index=df.index)
    
    for tf in timeframes:
        try:
            resampled = resample_data(df, tf)
            
            # Calculate pivot points
            calculate_pivot_points = _import_price_action()
            pivots = calculate_pivot_points(resampled, method='standard')
            
            # Map back to 1-minute
            resampled = resampled.set_index('date')
            pivots = pivots.set_index(resampled.index)
            pivots = pivots.reindex(df.index, method='ffill')
            
            result[f'support_{tf}'] = pivots[['s1', 's2', 's3']].min(axis=1)
            result[f'resistance_{tf}'] = pivots[['r1', 'r2', 'r3']].max(axis=1)
            
        except Exception as e:
            print(f"Warning: Could not process timeframe {tf}: {e}")
            result[f'support_{tf}'] = df['close']
            result[f'resistance_{tf}'] = df['close']
    
    return result


def get_momentum_multiple_timeframes(df: pd.DataFrame,
                                     timeframes: List[str] = ['15min', '1H', '1D']) -> pd.DataFrame:
    """
    Calculate momentum indicators across multiple timeframes.
    
    Args:
        df: DataFrame with 1-minute OHLCV data
        timeframes: List of timeframes to analyze
    
    Returns:
        DataFrame with momentum indicators
    """
    result = pd.DataFrame(index=df.index)
    
    for tf in timeframes:
        try:
            resampled = resample_data(df, tf)
            
            # Calculate RSI and MACD
            calculate_rsi, calculate_macd = _import_indicators()
            rsi = calculate_rsi(resampled, period=14)
            macd = calculate_macd(resampled)
            
            # Map back to 1-minute
            resampled = resampled.set_index('date')
            rsi = rsi.set_index(resampled.index)
            macd = macd.set_index(resampled.index)
            
            rsi = rsi.reindex(df.index, method='ffill')
            macd = macd.reindex(df.index, method='ffill')
            
            result[f'rsi_{tf}'] = rsi
            result[f'macd_{tf}'] = macd['macd']
            result[f'macd_signal_{tf}'] = macd['macd_signal']
            result[f'macd_hist_{tf}'] = macd['macd_histogram']
            
        except Exception as e:
            print(f"Warning: Could not process timeframe {tf}: {e}")
            result[f'rsi_{tf}'] = 50
            result[f'macd_{tf}'] = 0
            result[f'macd_signal_{tf}'] = 0
            result[f'macd_hist_{tf}'] = 0
    
    return result


def get_all_multiple_timeframe_signals(df: pd.DataFrame,
                                      timeframes: List[str] = ['15min', '1H', '1D'],
                                      config: Optional[dict] = None) -> pd.DataFrame:
    """
    Get all multiple timeframe analysis signals.
    
    Args:
        df: DataFrame with 1-minute OHLCV data
        timeframes: List of timeframes to analyze
        config: Optional configuration dict
    
    Returns:
        DataFrame with all multi-timeframe signals
    """
    result = pd.DataFrame(index=df.index)
    
    # Trend analysis
    trends = get_trend_multiple_timeframes(df, timeframes)
    result = pd.concat([result, trends], axis=1)
    
    # Support/Resistance
    sr = get_support_resistance_multiple_timeframes(df, timeframes)
    result = pd.concat([result, sr], axis=1)
    
    # Momentum
    momentum = get_momentum_multiple_timeframes(df, timeframes)
    result = pd.concat([result, momentum], axis=1)
    
    # Combined signals
    # Bullish: higher timeframes in uptrend, momentum positive
    bullish_trend_cols = [f'trend_{tf}' for tf in timeframes if f'trend_{tf}' in result.columns]
    if bullish_trend_cols:
        result['mtf_bullish'] = (result[bullish_trend_cols].sum(axis=1) > 0).astype(int)
        result['mtf_bearish'] = (result[bullish_trend_cols].sum(axis=1) < 0).astype(int)
    
    return result

