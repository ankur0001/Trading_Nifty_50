"""
Candlestick pattern recognition module.
Identifies single and multiple candlestick patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict


def identify_marubozu(df: pd.DataFrame) -> pd.Series:
    """
    Identify Marubozu patterns (strong bullish/bearish candles with no shadows).
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Series: 1 for bullish marubozu, -1 for bearish marubozu, 0 otherwise
    """
    body = abs(df['close'] - df['open'])
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    
    # Marubozu: body is large and shadows are very small
    body_ratio = body / (df['high'] - df['low'] + 1e-10)
    shadow_threshold = (df['high'] - df['low']) * 0.05
    
    bullish = (df['close'] > df['open']) & (body_ratio > 0.95) & \
              (upper_shadow < shadow_threshold) & (lower_shadow < shadow_threshold)
    bearish = (df['close'] < df['open']) & (body_ratio > 0.95) & \
              (upper_shadow < shadow_threshold) & (lower_shadow < shadow_threshold)
    
    result = pd.Series(0, index=df.index)
    result[bullish] = 1
    result[bearish] = -1
    
    return result


def identify_hammer(df: pd.DataFrame) -> pd.Series:
    """
    Identify Hammer patterns (bullish reversal at bottom).
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Series: 1 for hammer, 0 otherwise
    """
    body = abs(df['close'] - df['open'])
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    total_range = df['high'] - df['low']
    
    # Hammer: small body at top, long lower shadow, little/no upper shadow
    condition = (lower_shadow > 2 * body) & \
                (upper_shadow < body * 0.3) & \
                (total_range > 0) & \
                (lower_shadow > total_range * 0.6)
    
    return condition.astype(int)


def identify_hanging_man(df: pd.DataFrame) -> pd.Series:
    """
    Identify Hanging Man patterns (bearish reversal at top).
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Series: 1 for hanging man, 0 otherwise
    """
    body = abs(df['close'] - df['open'])
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    total_range = df['high'] - df['low']
    
    # Hanging Man: small body at top, long lower shadow, little/no upper shadow
    condition = (lower_shadow > 2 * body) & \
                (upper_shadow < body * 0.3) & \
                (total_range > 0) & \
                (lower_shadow > total_range * 0.6)
    
    return condition.astype(int)


def identify_shooting_star(df: pd.DataFrame) -> pd.Series:
    """
    Identify Shooting Star patterns (bearish reversal at top).
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Series: 1 for shooting star, 0 otherwise
    """
    body = abs(df['close'] - df['open'])
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    total_range = df['high'] - df['low']
    
    # Shooting Star: small body at bottom, long upper shadow, little/no lower shadow
    condition = (upper_shadow > 2 * body) & \
                (lower_shadow < body * 0.3) & \
                (total_range > 0) & \
                (upper_shadow > total_range * 0.6)
    
    return condition.astype(int)


def identify_doji(df: pd.DataFrame) -> pd.Series:
    """
    Identify Doji patterns (indecision, small body).
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Series: 1 for doji, 0 otherwise
    """
    body = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low']
    
    # Doji: very small body relative to range
    condition = (body < total_range * 0.1) & (total_range > 0)
    
    return condition.astype(int)


def identify_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Identify Engulfing patterns (bullish and bearish).
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Series: 1 for bullish engulfing, -1 for bearish engulfing, 0 otherwise
    """
    prev_body = abs(df['close'].shift(1) - df['open'].shift(1))
    curr_body = abs(df['close'] - df['open'])
    
    # Bullish Engulfing: current green candle engulfs previous red candle
    bullish = (df['close'] > df['open']) & \
              (df['open'].shift(1) > df['close'].shift(1)) & \
              (df['open'] < df['close'].shift(1)) & \
              (df['close'] > df['open'].shift(1)) & \
              (curr_body > prev_body)
    
    # Bearish Engulfing: current red candle engulfs previous green candle
    bearish = (df['close'] < df['open']) & \
              (df['open'].shift(1) < df['close'].shift(1)) & \
              (df['open'] > df['close'].shift(1)) & \
              (df['close'] < df['open'].shift(1)) & \
              (curr_body > prev_body)
    
    result = pd.Series(0, index=df.index)
    result[bullish] = 1
    result[bearish] = -1
    
    return result


def identify_piercing(df: pd.DataFrame) -> pd.Series:
    """
    Identify Piercing Pattern (bullish reversal).
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Series: 1 for piercing pattern, 0 otherwise
    """
    prev_close = df['close'].shift(1)
    prev_open = df['open'].shift(1)
    prev_body = prev_close - prev_open
    
    # Piercing: previous red candle, current green candle opens below prev close,
    # closes above midpoint of previous body
    midpoint = prev_open + (prev_body / 2)
    
    condition = (prev_body < 0) & \
                (df['close'] > df['open']) & \
                (df['open'] < prev_close) & \
                (df['close'] > midpoint) & \
                (df['close'] < prev_open)
    
    return condition.astype(int)


def identify_dark_cloud_cover(df: pd.DataFrame) -> pd.Series:
    """
    Identify Dark Cloud Cover pattern (bearish reversal).
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Series: 1 for dark cloud cover, 0 otherwise
    """
    prev_close = df['close'].shift(1)
    prev_open = df['open'].shift(1)
    prev_body = prev_close - prev_open
    
    # Dark Cloud Cover: previous green candle, current red candle opens above prev close,
    # closes below midpoint of previous body
    midpoint = prev_open + (prev_body / 2)
    
    condition = (prev_body > 0) & \
                (df['close'] < df['open']) & \
                (df['open'] > prev_close) & \
                (df['close'] < midpoint) & \
                (df['close'] > prev_open)
    
    return condition.astype(int)


def get_all_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify all candlestick patterns and return as DataFrame.
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        DataFrame with pattern columns
    """
    patterns = pd.DataFrame(index=df.index)
    
    patterns['marubozu'] = identify_marubozu(df)
    patterns['hammer'] = identify_hammer(df)
    patterns['hanging_man'] = identify_hanging_man(df)
    patterns['shooting_star'] = identify_shooting_star(df)
    patterns['doji'] = identify_doji(df)
    patterns['engulfing'] = identify_engulfing(df)
    patterns['piercing'] = identify_piercing(df)
    patterns['dark_cloud_cover'] = identify_dark_cloud_cover(df)
    
    # Calculate bullish/bearish signals
    patterns['bullish_signal'] = (
        (patterns['marubozu'] == 1) |
        (patterns['hammer'] == 1) |
        (patterns['engulfing'] == 1) |
        (patterns['piercing'] == 1)
    ).astype(int)
    
    patterns['bearish_signal'] = (
        (patterns['marubozu'] == -1) |
        (patterns['hanging_man'] == 1) |
        (patterns['shooting_star'] == 1) |
        (patterns['engulfing'] == -1) |
        (patterns['dark_cloud_cover'] == 1)
    ).astype(int)
    
    return patterns

