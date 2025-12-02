"""
Advanced technical indicators module.
Implements RSI, MACD, Chaikin Oscillator, Moving Averages, and more.
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_sma(df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        df: DataFrame with price data
        period: Period for SMA
        column: Column name to calculate SMA on
    
    Returns:
        Series with SMA values
    """
    return df[column].rolling(window=period, min_periods=1).mean()


def calculate_ema(df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        df: DataFrame with price data
        period: Period for EMA
        column: Column name to calculate EMA on
    
    Returns:
        Series with EMA values
    """
    return df[column].ewm(span=period, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        df: DataFrame with price data
        period: Period for RSI (default: 14)
        column: Column name to calculate RSI on
    
    Returns:
        Series with RSI values (0-100)
    """
    delta = df[column].diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, 
                   signal: int = 9, column: str = 'close') -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        df: DataFrame with price data
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line EMA period (default: 9)
        column: Column name to calculate MACD on
    
    Returns:
        DataFrame with MACD, Signal, and Histogram columns
    """
    ema_fast = calculate_ema(df, fast, column)
    ema_slow = calculate_ema(df, slow, column)
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    result = pd.DataFrame(index=df.index)
    result['macd'] = macd_line
    result['macd_signal'] = signal_line
    result['macd_histogram'] = histogram
    
    return result


def calculate_chaikin_oscillator(df: pd.DataFrame, fast: int = 3, 
                                slow: int = 10) -> pd.Series:
    """
    Calculate Chaikin Oscillator (Accumulation/Distribution Oscillator).
    
    Args:
        df: DataFrame with OHLCV data
        fast: Fast EMA period (default: 3)
        slow: Slow EMA period (default: 10)
    
    Returns:
        Series with Chaikin Oscillator values
    """
    # Calculate Money Flow Multiplier
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
          (df['high'] - df['low'] + 1e-10)
    
    # Calculate Accumulation/Distribution Line
    ad = (clv * df['volume']).cumsum()
    
    # Calculate Chaikin Oscillator (fast EMA - slow EMA of AD)
    fast_ema = ad.ewm(span=fast, adjust=False).mean()
    slow_ema = ad.ewm(span=slow, adjust=False).mean()
    
    chaikin = fast_ema - slow_ema
    
    return chaikin


def calculate_williams_fractals(df: pd.DataFrame, period: int = 2) -> pd.DataFrame:
    """
    Calculate Williams Fractals (identifies potential reversal points).
    
    Args:
        df: DataFrame with OHLC data
        period: Period for fractal detection (default: 2)
    
    Returns:
        DataFrame with fractal_up and fractal_down columns
    """
    result = pd.DataFrame(index=df.index)
    
    # Fractal Up: high is highest in the period
    fractal_up = pd.Series(False, index=df.index)
    fractal_down = pd.Series(False, index=df.index)
    
    for i in range(period, len(df) - period):
        # Check if current high is highest in the window
        window_highs = df['high'].iloc[i-period:i+period+1]
        if df['high'].iloc[i] == window_highs.max() and \
           df['high'].iloc[i] > df['high'].iloc[i-period] and \
           df['high'].iloc[i] > df['high'].iloc[i+period]:
            fractal_up.iloc[i] = True
        
        # Check if current low is lowest in the window
        window_lows = df['low'].iloc[i-period:i+period+1]
        if df['low'].iloc[i] == window_lows.min() and \
           df['low'].iloc[i] < df['low'].iloc[i-period] and \
           df['low'].iloc[i] < df['low'].iloc[i+period]:
            fractal_down.iloc[i] = True
    
    result['fractal_up'] = fractal_up.astype(int)
    result['fractal_down'] = fractal_down.astype(int)
    
    return result


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, 
                              std_dev: float = 2.0, column: str = 'close') -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Args:
        df: DataFrame with price data
        period: Period for moving average (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)
        column: Column name to calculate bands on
    
    Returns:
        DataFrame with upper, middle, and lower bands
    """
    sma = calculate_sma(df, period, column)
    std = df[column].rolling(window=period, min_periods=1).std()
    
    result = pd.DataFrame(index=df.index)
    result['bb_upper'] = sma + (std * std_dev)
    result['bb_middle'] = sma
    result['bb_lower'] = sma - (std * std_dev)
    result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
    result['bb_position'] = (df[column] - result['bb_lower']) / \
                           (result['bb_upper'] - result['bb_lower'] + 1e-10)
    
    return result


def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, 
                        d_period: int = 3) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        df: DataFrame with OHLC data
        k_period: Period for %K (default: 14)
        d_period: Period for %D (default: 3)
    
    Returns:
        DataFrame with %K and %D values
    """
    low_min = df['low'].rolling(window=k_period, min_periods=1).min()
    high_max = df['high'].rolling(window=k_period, min_periods=1).max()
    
    k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-10))
    d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()
    
    result = pd.DataFrame(index=df.index)
    result['stoch_k'] = k_percent
    result['stoch_d'] = d_percent
    
    return result


def get_all_indicators(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Calculate all technical indicators.
    
    Args:
        df: DataFrame with OHLCV data
        config: Optional configuration dict with indicator parameters
    
    Returns:
        DataFrame with all indicator columns
    """
    if config is None:
        config = {}
    
    indicators = pd.DataFrame(index=df.index)
    
    # Moving Averages
    indicators['sma_20'] = calculate_sma(df, config.get('sma_period', 20))
    indicators['sma_50'] = calculate_sma(df, config.get('sma_50_period', 50))
    indicators['ema_12'] = calculate_ema(df, config.get('ema_fast', 12))
    indicators['ema_26'] = calculate_ema(df, config.get('ema_slow', 26))
    
    # RSI
    indicators['rsi'] = calculate_rsi(df, config.get('rsi_period', 14))
    
    # MACD
    macd = calculate_macd(df, 
                         config.get('macd_fast', 12),
                         config.get('macd_slow', 26),
                         config.get('macd_signal', 9))
    indicators = pd.concat([indicators, macd], axis=1)
    
    # Chaikin Oscillator
    indicators['chaikin'] = calculate_chaikin_oscillator(df,
                                                        config.get('chaikin_fast', 3),
                                                        config.get('chaikin_slow', 10))
    
    # Williams Fractals
    fractals = calculate_williams_fractals(df, config.get('fractal_period', 2))
    indicators = pd.concat([indicators, fractals], axis=1)
    
    # Bollinger Bands
    bb = calculate_bollinger_bands(df, 
                                   config.get('bb_period', 20),
                                   config.get('bb_std', 2.0))
    indicators = pd.concat([indicators, bb], axis=1)
    
    # Stochastic
    stoch = calculate_stochastic(df,
                                 config.get('stoch_k', 14),
                                 config.get('stoch_d', 3))
    indicators = pd.concat([indicators, stoch], axis=1)
    
    return indicators

