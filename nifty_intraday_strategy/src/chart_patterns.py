"""
Chart pattern recognition module.
Identifies head & shoulders, double/triple tops/bottoms, and N-tops/bottoms.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
try:
    from scipy.signal import find_peaks
except ImportError:
    # Fallback if scipy not available
    def find_peaks(values, distance=1):
        peaks = []
        for i in range(distance, len(values) - distance):
            if values[i] == max(values[i-distance:i+distance+1]):
                peaks.append(i)
        return np.array(peaks), {}


def find_local_extrema(df: pd.DataFrame, lookback: int = 5, 
                      use_highs: bool = True) -> pd.Series:
    """
    Find local maxima (peaks) or minima (troughs).
    
    Args:
        df: DataFrame with price data
        lookback: Number of bars to look back/forward
        use_highs: True for peaks (highs), False for troughs (lows)
    
    Returns:
        Series with 1 at extrema, 0 otherwise
    """
    if use_highs:
        values = df['high'].values
    else:
        values = df['low'].values
    
    # Find peaks/troughs
    if use_highs:
        peaks, _ = find_peaks(values, distance=lookback)
    else:
        peaks, _ = find_peaks(-values, distance=lookback)
    
    result = pd.Series(0, index=df.index)
    result.iloc[peaks] = 1
    
    return result


def detect_double_top(df: pd.DataFrame, tolerance: float = 0.02, 
                     min_distance: int = 10, max_lookback: int = 500) -> pd.Series:
    """
    Detect Double Top pattern (optimized version).
    
    Args:
        df: DataFrame with OHLC data
        tolerance: Price tolerance for matching tops (default: 2%)
        min_distance: Minimum distance between tops in bars
        max_lookback: Maximum bars to look back (for performance)
    
    Returns:
        Series: 1 when double top detected, 0 otherwise
    """
    peaks = find_local_extrema(df, lookback=min_distance, use_highs=True)
    peak_indices = df.index[peaks == 1].tolist()
    
    result = pd.Series(0, index=df.index)
    
    if len(peak_indices) < 2:
        return result
    
    # Optimized: only check nearby peaks (within max_lookback)
    for i in range(len(peak_indices) - 1):
        # Limit lookback for performance
        start_j = max(0, i - max_lookback // min_distance) if i > max_lookback // min_distance else 0
        for j in range(max(i + 1, start_j), min(i + max_lookback // min_distance + 1, len(peak_indices))):
            idx1 = peak_indices[i]
            idx2 = peak_indices[j]
            
            # Skip if too far apart
            if abs(idx2 - idx1) > max_lookback:
                continue
            
            price1 = df.loc[idx1, 'high']
            price2 = df.loc[idx2, 'high']
            
            # Check if prices are similar and have a valley between them
            if abs(price1 - price2) / max(price1, price2) <= tolerance:
                # Check for valley between peaks
                valley_idx = df.loc[idx1:idx2, 'low'].idxmin()
                valley_price = df.loc[valley_idx, 'low']
                
                # Valley should be significantly lower
                if valley_price < min(price1, price2) * (1 - tolerance * 2):
                    # Mark the second peak as double top
                    result.loc[idx2] = 1
                    break  # Found a match, move to next peak
    
    return result


def detect_double_bottom(df: pd.DataFrame, tolerance: float = 0.02, 
                        min_distance: int = 10, max_lookback: int = 500) -> pd.Series:
    """
    Detect Double Bottom pattern (optimized version).
    
    Args:
        df: DataFrame with OHLC data
        tolerance: Price tolerance for matching bottoms (default: 2%)
        min_distance: Minimum distance between bottoms in bars
        max_lookback: Maximum bars to look back (for performance)
    
    Returns:
        Series: 1 when double bottom detected, 0 otherwise
    """
    troughs = find_local_extrema(df, lookback=min_distance, use_highs=False)
    trough_indices = df.index[troughs == 1].tolist()
    
    result = pd.Series(0, index=df.index)
    
    if len(trough_indices) < 2:
        return result
    
    # Optimized: only check nearby troughs (within max_lookback)
    for i in range(len(trough_indices) - 1):
        start_j = max(0, i - max_lookback // min_distance) if i > max_lookback // min_distance else 0
        for j in range(max(i + 1, start_j), min(i + max_lookback // min_distance + 1, len(trough_indices))):
            idx1 = trough_indices[i]
            idx2 = trough_indices[j]
            
            if abs(idx2 - idx1) > max_lookback:
                continue
            
            price1 = df.loc[idx1, 'low']
            price2 = df.loc[idx2, 'low']
            
            # Check if prices are similar and have a peak between them
            if abs(price1 - price2) / max(price1, price2) <= tolerance:
                # Check for peak between troughs
                peak_idx = df.loc[idx1:idx2, 'high'].idxmax()
                peak_price = df.loc[peak_idx, 'high']
                
                # Peak should be significantly higher
                if peak_price > max(price1, price2) * (1 + tolerance * 2):
                    # Mark the second trough as double bottom
                    result.loc[idx2] = 1
                    break
    
    return result


def detect_triple_top(df: pd.DataFrame, tolerance: float = 0.02, 
                     min_distance: int = 10, max_lookback: int = 500) -> pd.Series:
    """
    Detect Triple Top pattern (optimized version).
    
    Args:
        df: DataFrame with OHLC data
        tolerance: Price tolerance for matching tops
        min_distance: Minimum distance between tops
        max_lookback: Maximum bars to look back (for performance)
    
    Returns:
        Series: 1 when triple top detected, 0 otherwise
    """
    peaks = find_local_extrema(df, lookback=min_distance, use_highs=True)
    peak_indices = df.index[peaks == 1].tolist()
    
    result = pd.Series(0, index=df.index)
    
    if len(peak_indices) < 3:
        return result
    
    # Optimized: limit search window
    max_window = max_lookback // min_distance
    for i in range(len(peak_indices) - 2):
        for j in range(i + 1, min(i + max_window + 1, len(peak_indices) - 1)):
            idx1 = peak_indices[i]
            idx2 = peak_indices[j]
            if abs(idx2 - idx1) > max_lookback:
                continue
            for k in range(j + 1, min(j + max_window + 1, len(peak_indices))):
                idx3 = peak_indices[k]
                if abs(idx3 - idx1) > max_lookback:
                    continue
                
                price1 = df.loc[idx1, 'high']
                price2 = df.loc[idx2, 'high']
                price3 = df.loc[idx3, 'high']
                
                avg_price = (price1 + price2 + price3) / 3
                
                # Check if all three prices are similar
                if all(abs(p - avg_price) / avg_price <= tolerance for p in [price1, price2, price3]):
                    result.loc[idx3] = 1
                    break
    
    return result


def detect_triple_bottom(df: pd.DataFrame, tolerance: float = 0.02, 
                        min_distance: int = 10, max_lookback: int = 500) -> pd.Series:
    """
    Detect Triple Bottom pattern.
    
    Args:
        df: DataFrame with OHLC data
        tolerance: Price tolerance for matching bottoms
        min_distance: Minimum distance between bottoms
    
    Returns:
        Series: 1 when triple bottom detected, 0 otherwise
    """
    troughs = find_local_extrema(df, lookback=min_distance, use_highs=False)
    trough_indices = df.index[troughs == 1].tolist()
    
    result = pd.Series(0, index=df.index)
    
    if len(trough_indices) < 3:
        return result
    
    # Optimized: limit search window
    max_window = max_lookback // min_distance
    for i in range(len(trough_indices) - 2):
        for j in range(i + 1, min(i + max_window + 1, len(trough_indices) - 1)):
            idx1 = trough_indices[i]
            idx2 = trough_indices[j]
            if abs(idx2 - idx1) > max_lookback:
                continue
            for k in range(j + 1, min(j + max_window + 1, len(trough_indices))):
                idx3 = trough_indices[k]
                if abs(idx3 - idx1) > max_lookback:
                    continue
                
                price1 = df.loc[idx1, 'low']
                price2 = df.loc[idx2, 'low']
                price3 = df.loc[idx3, 'low']
                
                avg_price = (price1 + price2 + price3) / 3
                
                # Check if all three prices are similar
                if all(abs(p - avg_price) / avg_price <= tolerance for p in [price1, price2, price3]):
                    result.loc[idx3] = 1
                    break
    
    return result


def detect_head_and_shoulders(df: pd.DataFrame, tolerance: float = 0.02, 
                             min_distance: int = 10, max_lookback: int = 500) -> pd.Series:
    """
    Detect Head and Shoulders pattern.
    
    Args:
        df: DataFrame with OHLC data
        tolerance: Price tolerance for matching shoulders
        min_distance: Minimum distance between peaks
    
    Returns:
        Series: 1 when H&S detected, 0 otherwise
    """
    peaks = find_local_extrema(df, lookback=min_distance, use_highs=True)
    peak_indices = df.index[peaks == 1].tolist()
    
    result = pd.Series(0, index=df.index)
    
    if len(peak_indices) < 3:
        return result
    
    # Look for pattern: left shoulder, head (higher), right shoulder (similar to left)
    for i in range(len(peak_indices) - 2):
        left_shoulder_idx = peak_indices[i]
        head_idx = peak_indices[i + 1]
        right_shoulder_idx = peak_indices[i + 2]
        
        left_price = df.loc[left_shoulder_idx, 'high']
        head_price = df.loc[head_idx, 'high']
        right_price = df.loc[right_shoulder_idx, 'high']
        
        # Head should be higher than shoulders
        if head_price > left_price and head_price > right_price:
            # Shoulders should be similar
            if abs(left_price - right_price) / max(left_price, right_price) <= tolerance:
                # Check for neckline (valley between left shoulder and head, and between head and right shoulder)
                valley1_idx = df.loc[left_shoulder_idx:head_idx, 'low'].idxmin()
                valley2_idx = df.loc[head_idx:right_shoulder_idx, 'low'].idxmin()
                
                valley1_price = df.loc[valley1_idx, 'low']
                valley2_price = df.loc[valley2_idx, 'low']
                
                # Neckline should be similar
                if abs(valley1_price - valley2_price) / max(valley1_price, valley2_price) <= tolerance:
                    result.loc[right_shoulder_idx] = 1
    
    return result


def detect_inverse_head_and_shoulders(df: pd.DataFrame, tolerance: float = 0.02, 
                                     min_distance: int = 10, max_lookback: int = 500) -> pd.Series:
    """
    Detect Inverse Head and Shoulders pattern.
    
    Args:
        df: DataFrame with OHLC data
        tolerance: Price tolerance for matching shoulders
        min_distance: Minimum distance between troughs
    
    Returns:
        Series: 1 when inverse H&S detected, 0 otherwise
    """
    troughs = find_local_extrema(df, lookback=min_distance, use_highs=False)
    trough_indices = df.index[troughs == 1].tolist()
    
    result = pd.Series(0, index=df.index)
    
    if len(trough_indices) < 3:
        return result
    
    # Look for pattern: left shoulder, head (lower), right shoulder (similar to left)
    # Optimized: limit search window
    max_window = min(max_lookback // min_distance, len(trough_indices) - 2)
    for i in range(min(max_window, len(trough_indices) - 2)):
        left_shoulder_idx = trough_indices[i]
        if i + 2 >= len(trough_indices):
            break
        head_idx = trough_indices[i + 1]
        right_shoulder_idx = trough_indices[i + 2]
        
        # Skip if too far apart
        if abs(right_shoulder_idx - left_shoulder_idx) > max_lookback:
            continue
        
        left_price = df.loc[left_shoulder_idx, 'low']
        head_price = df.loc[head_idx, 'low']
        right_price = df.loc[right_shoulder_idx, 'low']
        
        # Head should be lower than shoulders
        if head_price < left_price and head_price < right_price:
            # Shoulders should be similar
            if abs(left_price - right_price) / max(left_price, right_price) <= tolerance:
                # Check for neckline (peaks between shoulders and head)
                peak1_idx = df.loc[left_shoulder_idx:head_idx, 'high'].idxmax()
                peak2_idx = df.loc[head_idx:right_shoulder_idx, 'high'].idxmax()
                
                peak1_price = df.loc[peak1_idx, 'high']
                peak2_price = df.loc[peak2_idx, 'high']
                
                # Neckline should be similar
                if abs(peak1_price - peak2_price) / max(peak1_price, peak2_price) <= tolerance:
                    result.loc[right_shoulder_idx] = 1
    
    return result


def detect_n_tops_bottoms(df: pd.DataFrame, n: int = 4, tolerance: float = 0.02, 
                          min_distance: int = 10, use_highs: bool = True) -> pd.Series:
    """
    Detect N-tops or N-bottoms pattern.
    
    Args:
        df: DataFrame with OHLC data
        n: Number of tops/bottoms to detect
        tolerance: Price tolerance for matching
        min_distance: Minimum distance between extrema
        use_highs: True for N-tops, False for N-bottoms
    
    Returns:
        Series: 1 when N-pattern detected, 0 otherwise
    """
    extrema = find_local_extrema(df, lookback=min_distance, use_highs=use_highs)
    extrema_indices = df.index[extrema == 1].tolist()
    
    result = pd.Series(0, index=df.index)
    
    if len(extrema_indices) < n:
        return result
    
    # Check for n similar extrema
    for i in range(len(extrema_indices) - n + 1):
        indices = extrema_indices[i:i+n]
        
        if use_highs:
            prices = [df.loc[idx, 'high'] for idx in indices]
        else:
            prices = [df.loc[idx, 'low'] for idx in indices]
        
        avg_price = np.mean(prices)
        max_deviation = max(abs(p - avg_price) / avg_price for p in prices)
        
        if max_deviation <= tolerance:
            result.loc[indices[-1]] = 1
    
    return result


def get_all_chart_patterns(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Detect all chart patterns (optimized version with progress tracking).
    
    Args:
        df: DataFrame with OHLC data
        config: Optional configuration dict
    
    Returns:
        DataFrame with pattern columns
    """
    if config is None:
        config = {}
    
    import sys
    patterns = pd.DataFrame(index=df.index)
    
    tolerance = config.get('pattern_tolerance', 0.02)
    min_distance = config.get('pattern_min_distance', 10)
    max_lookback = config.get('pattern_max_lookback', 500)  # Limit lookback for performance
    
    print("  Detecting double top...", end='', flush=True)
    patterns['double_top'] = detect_double_top(df, tolerance, min_distance, max_lookback)
    print(" ✓")
    
    print("  Detecting double bottom...", end='', flush=True)
    patterns['double_bottom'] = detect_double_bottom(df, tolerance, min_distance, max_lookback)
    print(" ✓")
    
    print("  Detecting triple top...", end='', flush=True)
    patterns['triple_top'] = detect_triple_top(df, tolerance, min_distance, max_lookback)
    print(" ✓")
    
    print("  Detecting triple bottom...", end='', flush=True)
    patterns['triple_bottom'] = detect_triple_bottom(df, tolerance, min_distance, max_lookback)
    print(" ✓")
    
    print("  Detecting head & shoulders...", end='', flush=True)
    patterns['head_shoulders'] = detect_head_and_shoulders(df, tolerance, min_distance, max_lookback)
    print(" ✓")
    
    print("  Detecting inverse head & shoulders...", end='', flush=True)
    patterns['inverse_head_shoulders'] = detect_inverse_head_and_shoulders(df, tolerance, min_distance, max_lookback)
    print(" ✓")
    
    # N-tops and N-bottoms (4-tops, 4-bottoms) - skip for performance
    # patterns['n_top_4'] = detect_n_tops_bottoms(df, n=4, tolerance=tolerance, 
    #                                             min_distance=min_distance, use_highs=True)
    # patterns['n_bottom_4'] = detect_n_tops_bottoms(df, n=4, tolerance=tolerance, 
    #                                                min_distance=min_distance, use_highs=False)
    patterns['n_top_4'] = pd.Series(0, index=df.index)
    patterns['n_bottom_4'] = pd.Series(0, index=df.index)
    
    # Summary signals
    patterns['bearish_pattern'] = (
        (patterns['double_top'] == 1) |
        (patterns['triple_top'] == 1) |
        (patterns['head_shoulders'] == 1) |
        (patterns['n_top_4'] == 1)
    ).astype(int)
    
    patterns['bullish_pattern'] = (
        (patterns['double_bottom'] == 1) |
        (patterns['triple_bottom'] == 1) |
        (patterns['inverse_head_shoulders'] == 1) |
        (patterns['n_bottom_4'] == 1)
    ).astype(int)
    
    return patterns

