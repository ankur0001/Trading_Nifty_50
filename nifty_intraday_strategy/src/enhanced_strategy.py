"""
Enhanced trading strategy combining all indicators, patterns, and analysis tools.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import time
import sys
import os

# Handle imports for both module and direct execution
try:
    from .strategy import ORBStrategy
    from .indicators import calculate_vwap, calculate_atr
    from .candlestick_patterns import get_all_candlestick_patterns
    from .advanced_indicators import get_all_indicators
    from .chart_patterns import get_all_chart_patterns
    from .price_action import get_all_price_action
    from .volatility import get_all_volatility_measures
    from .multi_timeframe import get_all_multiple_timeframe_signals
except ImportError:
    from strategy import ORBStrategy
    from indicators import calculate_vwap, calculate_atr
    from candlestick_patterns import get_all_candlestick_patterns
    from advanced_indicators import get_all_indicators
    from chart_patterns import get_all_chart_patterns
    from price_action import get_all_price_action
    from volatility import get_all_volatility_measures
    from multi_timeframe import get_all_multiple_timeframe_signals


class EnhancedStrategy(ORBStrategy):
    """
    Enhanced strategy that combines:
    - Opening Range Breakout
    - Candlestick patterns
    - Technical indicators (RSI, MACD, Chaikin, etc.)
    - Chart patterns
    - Price action tools
    - Volatility measures
    - Multiple timeframe analysis
    """
    
    def __init__(self, config: Dict):
        """
        Initialize enhanced strategy.
        
        Args:
            config: Dictionary with strategy parameters
        """
        super().__init__(config)
        
        # Store config for indicator calculations
        self.config = config
        
        # Enhanced parameters
        self.use_candlestick = config.get('use_candlestick', True)
        self.use_indicators = config.get('use_indicators', True)
        self.use_chart_patterns = config.get('use_chart_patterns', True)
        self.use_price_action = config.get('use_price_action', True)
        self.use_volatility = config.get('use_volatility', True)
        self.use_multiple_timeframe = config.get('use_multiple_timeframe', True)
        
        # Indicator thresholds
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.macd_threshold = config.get('macd_threshold', 0)
        
        # Multi-timeframe settings
        self.mtf_timeframes = config.get('mtf_timeframes', ['15min', '1H', '1D'])
        
        # Store calculated indicators
        self.indicators_df = None
        self.patterns_df = None
        self.price_action_df = None
        self.volatility_df = None
        self.mtf_df = None
        
        # Cache manager
        self.use_cache = config.get('use_cache', True)
        self.cache_manager = None
        if self.use_cache:
            try:
                from .cache_manager import CacheManager
                cache_dir = config.get('cache_dir', 'cache')
                self.cache_manager = CacheManager(cache_dir)
            except ImportError:
                try:
                    from cache_manager import CacheManager
                    cache_dir = config.get('cache_dir', 'cache')
                    self.cache_manager = CacheManager(cache_dir)
                except ImportError:
                    self.cache_manager = None
                    print("Warning: Cache manager not available, caching disabled")
    
    def calculate_all_indicators(self, df: pd.DataFrame):
        """
        Calculate all indicators and patterns for the dataset (with caching).
        
        Args:
            df: DataFrame with OHLCV data
        """
        # Candlestick patterns
        if self.use_candlestick:
            print("Calculating candlestick patterns...", end='', flush=True)
            if self.cache_manager:
                cache_key = self.cache_manager.get_cache_key(df, self.config, 'candlestick_patterns')
                cached = self.cache_manager.load_from_cache(cache_key, 'candlestick_patterns')
                if cached is not None:
                    print(" ✓ (from cache)")
                    self.patterns_df = cached
                else:
                    self.patterns_df = get_all_candlestick_patterns(df)
                    self.cache_manager.save_to_cache(self.patterns_df, cache_key, 'candlestick_patterns')
                    print(" ✓")
            else:
                self.patterns_df = get_all_candlestick_patterns(df)
                print(" ✓")
        else:
            self.patterns_df = pd.DataFrame(index=df.index)
        
        # Technical indicators
        if self.use_indicators:
            print("Calculating technical indicators...", end='', flush=True)
            if self.cache_manager:
                cache_key = self.cache_manager.get_cache_key(df, self.config, 'technical_indicators')
                cached = self.cache_manager.load_from_cache(cache_key, 'technical_indicators')
                if cached is not None:
                    print(" ✓ (from cache)")
                    self.indicators_df = cached
                else:
                    self.indicators_df = get_all_indicators(df, self.config)
                    self.cache_manager.save_to_cache(self.indicators_df, cache_key, 'technical_indicators')
                    print(" ✓")
            else:
                self.indicators_df = get_all_indicators(df, self.config)
                print(" ✓")
        else:
            self.indicators_df = pd.DataFrame(index=df.index)
        
        # Chart patterns (optional, can be slow)
        if self.use_chart_patterns:
            print("Calculating chart patterns...")
            if self.cache_manager:
                cache_key = self.cache_manager.get_cache_key(df, self.config, 'chart_patterns')
                cached = self.cache_manager.load_from_cache(cache_key, 'chart_patterns')
                if cached is not None:
                    print("  ✓ Loaded from cache")
                    chart_patterns = cached
                else:
                    chart_patterns = get_all_chart_patterns(df, self.config)
                    self.cache_manager.save_to_cache(chart_patterns, cache_key, 'chart_patterns')
            else:
                chart_patterns = get_all_chart_patterns(df, self.config)
            
            if self.patterns_df is not None:
                self.patterns_df = pd.concat([self.patterns_df, chart_patterns], axis=1)
            else:
                self.patterns_df = chart_patterns
        else:
            print("Skipping chart patterns (disabled in config)")
        
        # Price action tools
        if self.use_price_action:
            print("Calculating price action tools...", end='', flush=True)
            if self.cache_manager:
                cache_key = self.cache_manager.get_cache_key(df, self.config, 'price_action')
                cached = self.cache_manager.load_from_cache(cache_key, 'price_action')
                if cached is not None:
                    print(" ✓ (from cache)")
                    self.price_action_df = cached
                else:
                    self.price_action_df = get_all_price_action(df, self.config)
                    self.cache_manager.save_to_cache(self.price_action_df, cache_key, 'price_action')
                    print(" ✓")
            else:
                self.price_action_df = get_all_price_action(df, self.config)
                print(" ✓")
        else:
            self.price_action_df = pd.DataFrame(index=df.index)
        
        # Volatility measures
        if self.use_volatility:
            print("Calculating volatility measures...", end='', flush=True)
            if self.cache_manager:
                cache_key = self.cache_manager.get_cache_key(df, self.config, 'volatility')
                cached = self.cache_manager.load_from_cache(cache_key, 'volatility')
                if cached is not None:
                    print(" ✓ (from cache)")
                    self.volatility_df = cached
                else:
                    self.volatility_df = get_all_volatility_measures(df, self.config)
                    self.cache_manager.save_to_cache(self.volatility_df, cache_key, 'volatility')
                    print(" ✓")
            else:
                self.volatility_df = get_all_volatility_measures(df, self.config)
                print(" ✓")
        else:
            self.volatility_df = pd.DataFrame(index=df.index)
        
        # Multiple timeframe (can be slow)
        if self.use_multiple_timeframe:
            print("Calculating multiple timeframe signals...", end='', flush=True)
            if self.cache_manager:
                mtf_params = {**self.config, 'mtf_timeframes': self.mtf_timeframes}
                cache_key = self.cache_manager.get_cache_key(df, mtf_params, 'multi_timeframe')
                cached = self.cache_manager.load_from_cache(cache_key, 'multi_timeframe')
                if cached is not None:
                    print(" ✓ (from cache)")
                    self.mtf_df = cached
                else:
                    self.mtf_df = get_all_multiple_timeframe_signals(df, self.mtf_timeframes, self.config)
                    self.cache_manager.save_to_cache(self.mtf_df, cache_key, 'multi_timeframe')
                    print(" ✓")
            else:
                self.mtf_df = get_all_multiple_timeframe_signals(df, self.mtf_timeframes, self.config)
                print(" ✓")
        else:
            self.mtf_df = pd.DataFrame(index=df.index)
    
    def check_enhanced_entry_conditions(self, row: pd.Series, or_high: float, 
                                       or_low: float, vwap: float, 
                                       idx: int) -> Optional[str]:
        """
        Check enhanced entry conditions with all indicators.
        
        Args:
            row: Current bar data
            or_high: Opening range high
            or_low: Opening range low
            vwap: Current VWAP value
            idx: Current index in dataframe
        
        Returns:
            'LONG', 'SHORT', or None
        """
        if self.trades_today >= self.max_trades_per_day:
            return None
        
        if self.current_position is not None:
            return None
        
        close = row['close']
        bullish_score = 0
        bearish_score = 0
        
        # Base ORB conditions
        long_orb = close > or_high and close > vwap
        short_orb = close < or_low and close < vwap
        
        if not (long_orb or short_orb):
            return None
        
        # Candlestick patterns
        if self.use_candlestick and self.patterns_df is not None and idx < len(self.patterns_df):
            if self.patterns_df.iloc[idx]['bullish_signal'] == 1:
                bullish_score += 2
            if self.patterns_df.iloc[idx]['bearish_signal'] == 1:
                bearish_score += 2
        
        # Technical indicators
        if self.use_indicators and self.indicators_df is not None and idx < len(self.indicators_df):
            ind_row = self.indicators_df.iloc[idx]
            
            # RSI
            if not pd.isna(ind_row.get('rsi', 50)):
                rsi = ind_row['rsi']
                if rsi < self.rsi_oversold:
                    bullish_score += 1
                elif rsi > self.rsi_overbought:
                    bearish_score += 1
            
            # MACD
            if not pd.isna(ind_row.get('macd_histogram', 0)):
                macd_hist = ind_row['macd_histogram']
                if macd_hist > self.macd_threshold:
                    bullish_score += 1
                elif macd_hist < -self.macd_threshold:
                    bearish_score += 1
            
            # Moving averages
            if not pd.isna(ind_row.get('sma_20', 0)) and not pd.isna(ind_row.get('sma_50', 0)):
                if close > ind_row['sma_20'] > ind_row['sma_50']:
                    bullish_score += 1
                elif close < ind_row['sma_20'] < ind_row['sma_50']:
                    bearish_score += 1
            
            # Chaikin Oscillator
            if not pd.isna(ind_row.get('chaikin', 0)):
                if ind_row['chaikin'] > 0:
                    bullish_score += 1
                else:
                    bearish_score += 1
        
        # Chart patterns
        if self.use_chart_patterns and self.patterns_df is not None and idx < len(self.patterns_df):
            if self.patterns_df.iloc[idx].get('bullish_pattern', 0) == 1:
                bullish_score += 2
            if self.patterns_df.iloc[idx].get('bearish_pattern', 0) == 1:
                bearish_score += 2
        
        # Price action (support/resistance)
        if self.use_price_action and self.price_action_df is not None and idx < len(self.price_action_df):
            pa_row = self.price_action_df.iloc[idx]
            if not pd.isna(pa_row.get('near_support', 0)) and pa_row['near_support'] == 1:
                bullish_score += 1
            if not pd.isna(pa_row.get('near_resistance', 0)) and pa_row['near_resistance'] == 1:
                bearish_score += 1
        
        # Multiple timeframe
        if self.use_multiple_timeframe and self.mtf_df is not None and idx < len(self.mtf_df):
            mtf_row = self.mtf_df.iloc[idx]
            if not pd.isna(mtf_row.get('mtf_bullish', 0)) and mtf_row['mtf_bullish'] == 1:
                bullish_score += 2
            if not pd.isna(mtf_row.get('mtf_bearish', 0)) and mtf_row['mtf_bearish'] == 1:
                bearish_score += 2
        
        # Volatility filter (avoid trading in extreme volatility)
        if self.use_volatility and self.volatility_df is not None and idx < len(self.volatility_df):
            vol_row = self.volatility_df.iloc[idx]
            if not pd.isna(vol_row.get('volatility_regime', 0)):
                if vol_row['volatility_regime'] == 1:  # High volatility
                    # Reduce scores in high volatility
                    bullish_score *= 0.7
                    bearish_score *= 0.7
        
        # Decision logic
        # Require minimum score and ORB condition
        min_score = 2
        
        if long_orb and bullish_score >= min_score and bullish_score > bearish_score:
            return 'LONG'
        
        if short_orb and bearish_score >= min_score and bearish_score > bullish_score:
            return 'SHORT'
        
        return None
    
    def check_entry_conditions(self, row: pd.Series, or_high: float, or_low: float, 
                              vwap: float) -> Optional[str]:
        """
        Override base class method to use enhanced conditions.
        Note: This requires idx which we'll handle in the backtester.
        """
        # This will be called from backtester with proper context
        return super().check_entry_conditions(row, or_high, or_low, vwap)

