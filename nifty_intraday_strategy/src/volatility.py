"""
Volatility measurement module.
Implements ATR, Standard Deviation, Beta, and other volatility metrics.
"""

import pandas as pd
import numpy as np
from typing import Optional

# Handle imports for both module and direct execution
try:
    from .indicators import calculate_atr
except ImportError:
    from indicators import calculate_atr


def calculate_volatility_std(df: pd.DataFrame, period: int = 20, 
                            column: str = 'close') -> pd.Series:
    """
    Calculate volatility using Standard Deviation of returns.
    
    Args:
        df: DataFrame with price data
        period: Period for calculation
        column: Column name to calculate on
    
    Returns:
        Series with volatility (standard deviation of returns)
    """
    returns = df[column].pct_change()
    volatility = returns.rolling(window=period, min_periods=1).std()
    
    # Annualize if needed (assuming daily data)
    # For intraday, this would need adjustment based on timeframe
    return volatility


def calculate_volatility_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate volatility using Average True Range (ATR).
    
    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation
    
    Returns:
        Series with ATR values
    """
    return calculate_atr(df, period)


def calculate_volatility_percent(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate ATR as percentage of price (normalized volatility).
    
    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation
    
    Returns:
        Series with ATR percentage
    """
    atr = calculate_atr(df, period)
    atr_pct = (atr / df['close']) * 100
    return atr_pct


def calculate_beta(df: pd.DataFrame, market_returns: pd.Series, 
                  period: int = 20) -> pd.Series:
    """
    Calculate Beta (sensitivity to market movements).
    
    Args:
        df: DataFrame with price data
        market_returns: Series with market returns
        period: Period for beta calculation
    
    Returns:
        Series with beta values
    """
    asset_returns = df['close'].pct_change()
    
    # Align indices
    aligned_returns = pd.DataFrame({
        'asset': asset_returns,
        'market': market_returns
    }).dropna()
    
    beta = pd.Series(index=df.index, dtype=float)
    
    for i in range(period, len(aligned_returns)):
        window_asset = aligned_returns['asset'].iloc[i-period:i]
        window_market = aligned_returns['market'].iloc[i-period:i]
        
        if window_market.std() > 0:
            cov = window_asset.cov(window_market)
            var_market = window_market.var()
            beta.iloc[i] = cov / var_market if var_market > 0 else 0
        else:
            beta.iloc[i] = 0
    
    return beta


def calculate_parkinson_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Parkinson Volatility (uses high-low range).
    More efficient estimator than close-to-close.
    
    Args:
        df: DataFrame with OHLC data
        period: Period for calculation
    
    Returns:
        Series with Parkinson volatility
    """
    # Parkinson estimator: sqrt((1/(4*ln(2))) * mean((ln(H/L))^2))
    log_hl = np.log(df['high'] / df['low'])
    parkinson = np.sqrt((1 / (4 * np.log(2))) * 
                       log_hl.rolling(window=period, min_periods=1).mean())
    
    return parkinson


def calculate_garman_klass_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Garman-Klass Volatility (uses OHLC data).
    More efficient than close-to-close volatility.
    
    Args:
        df: DataFrame with OHLC data
        period: Period for calculation
    
    Returns:
        Series with Garman-Klass volatility
    """
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])
    
    # Garman-Klass estimator
    gk = np.sqrt(0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2)
    gk_vol = gk.rolling(window=period, min_periods=1).mean()
    
    return gk_vol


def calculate_volatility_ratio(df: pd.DataFrame, short_period: int = 10, 
                               long_period: int = 20) -> pd.Series:
    """
    Calculate Volatility Ratio (short-term / long-term volatility).
    High ratio indicates increasing volatility.
    
    Args:
        df: DataFrame with price data
        short_period: Short-term period
        long_period: Long-term period
    
    Returns:
        Series with volatility ratio
    """
    short_vol = calculate_volatility_std(df, short_period)
    long_vol = calculate_volatility_std(df, long_period)
    
    ratio = short_vol / (long_vol + 1e-10)
    
    return ratio


def get_all_volatility_measures(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Calculate all volatility measures.
    
    Args:
        df: DataFrame with OHLC data
        config: Optional configuration dict
    
    Returns:
        DataFrame with all volatility measures
    """
    if config is None:
        config = {}
    
    volatility = pd.DataFrame(index=df.index)
    
    period = config.get('volatility_period', 20)
    atr_period = config.get('atr_period', 14)
    
    # Standard deviation volatility
    volatility['volatility_std'] = calculate_volatility_std(df, period)
    
    # ATR-based volatility
    volatility['volatility_atr'] = calculate_volatility_atr(df, atr_period)
    volatility['volatility_atr_pct'] = calculate_volatility_percent(df, atr_period)
    
    # Advanced volatility estimators
    volatility['volatility_parkinson'] = calculate_parkinson_volatility(df, period)
    volatility['volatility_gk'] = calculate_garman_klass_volatility(df, period)
    
    # Volatility ratio
    volatility['volatility_ratio'] = calculate_volatility_ratio(df,
                                                                config.get('vol_short', 10),
                                                                config.get('vol_long', 20))
    
    # Volatility regime (high/low)
    volatility['volatility_regime'] = (volatility['volatility_ratio'] > 1.2).astype(int)
    
    return volatility

