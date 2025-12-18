"""
Advanced Trading Strategies for NIFTY - 1m/5m Timeframes
High Potential Strategies Worth Backtesting

This module contains quant-friendly strategies that have shown promise:
1. Holy Grail Strategy (EMA pullback in strong trend)
3. Triangle Breakout with Confirmation
4. Range Breakout of 3rd 5-min Candle
8. Gap Open Strategy (2nd 5m candle)
9. Big Candle Breakout Strategy
10. Flag Breakout
11. CPR + 20 EMA Strategy
12. 15-Minute Breakout Strategy
15. Last 30-min Range of Previous Day
17. 20 EMA Mean Reversion
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import time, timedelta
import sys
import os

# Handle imports
try:
    from .indicators import calculate_vwap, calculate_atr
    from .advanced_indicators import get_all_indicators
except ImportError:
    from indicators import calculate_vwap, calculate_atr
    from advanced_indicators import get_all_indicators


class AdvancedStrategyBase:
    """Base class for advanced strategies."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.start_time = pd.to_datetime(config['start_time']).time()
        self.end_time = pd.to_datetime(config['end_time']).time()
        self.max_trades_per_day = config.get('max_trades_per_day', 1)
        
        # Track state
        self.current_position = None
        self.entry_price = None
        self.entry_time = None
        self.stop_loss = None
        self.trades_today = 0
        self.current_date = None
        
        # Store calculated indicators
        self.indicators_df = None
        
    def reset_daily_state(self, current_date):
        """Reset daily state variables."""
        if self.current_date != current_date:
            self.current_date = current_date
            self.trades_today = 0
            if self.current_position is not None:
                self.current_position = None
                self.entry_price = None
                self.entry_time = None
                self.stop_loss = None
    
    def calculate_stop_loss(self, entry_price: float, side: str, atr: float) -> float:
        """Calculate stop loss price based on ATR."""
        atr_mult = self.config.get('atr_mult', 1.5)
        if side == 'LONG':
            return entry_price - (atr_mult * atr)
        else:
            return entry_price + (atr_mult * atr)
    
    def enter_position(self, side: str, entry_price: float, entry_time: pd.Timestamp, atr: float):
        """Enter a new position."""
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


class HolyGrailStrategy(AdvancedStrategyBase):
    """
    Strategy 1: Holy Grail Strategy (EMA pullback in strong trend)
    
    Requirements:
    - 20 EMA > 50 EMA (uptrend) or 20 EMA < 50 EMA (downtrend)
    - ADX > 20 (strong trend)
    - VWAP in trend direction
    - ATR expanding
    - Entry on EMA pullback
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.adx_threshold = config.get('adx_threshold', 20)
        self.ema_fast = config.get('ema_fast', 20)
        self.ema_slow = config.get('ema_slow', 50)
    
    def check_entry_conditions(self, row: pd.Series, idx: int, df: pd.DataFrame) -> Optional[str]:
        """Check if entry conditions are met."""
        if self.trades_today >= self.max_trades_per_day:
            return None
        if self.current_position is not None:
            return None
        
        if idx < self.ema_slow:
            return None
        
        # Get indicators
        ema20 = df['ema_20'].iloc[idx] if 'ema_20' in df.columns else None
        ema50 = df['ema_50'].iloc[idx] if 'ema_50' in df.columns else None
        # ADX might not be available, skip if not present
        adx = df['adx'].iloc[idx] if 'adx' in df.columns else None
        vwap = row.get('vwap', None)
        atr = row.get('atr', None)
        close = row['close']
        
        # Check for None or NaN values
        if any(x is None or pd.isna(x) for x in [ema20, ema50, vwap, atr]):
            return None
        
        # ADX is optional but preferred
        if adx is None or pd.isna(adx):
            # If ADX not available, skip this strategy (it's critical)
            return None
        
        # Check trend direction
        uptrend = ema20 > ema50
        downtrend = ema20 < ema50
        
        if not (uptrend or downtrend):
            return None
        
        # ADX must be > threshold
        if adx <= self.adx_threshold:
            return None
        
        # VWAP must be in trend direction
        if uptrend and close < vwap:
            return None
        if downtrend and close > vwap:
            return None
        
        # Check ATR expansion (current ATR > previous 5-period average)
        if idx >= 5:
            atr_avg = df['atr'].iloc[idx-5:idx].mean()
            if atr <= atr_avg:
                return None
        
        # Check for pullback to EMA
        if uptrend:
            # Price pulled back to EMA20 but didn't break below
            if row['low'] <= ema20 <= row['high'] and close > ema20:
                return 'LONG'
        elif downtrend:
            # Price pulled back to EMA20 but didn't break above
            if row['low'] <= ema20 <= row['high'] and close < ema20:
                return 'SHORT'
        
        return None
    
    def check_exit_conditions(self, row: pd.Series, vwap: float, current_time: pd.Timestamp) -> bool:
        """Check if exit conditions are met."""
        if self.current_position is None:
            return False
        
        # Stop loss
        if self.current_position == 'LONG':
            if row['low'] <= self.stop_loss:
                return True
            # Exit if VWAP cross back
            if row['close'] < vwap:
                return True
        else:  # SHORT
            if row['high'] >= self.stop_loss:
                return True
            if row['close'] > vwap:
                return True
        
        # End of day
        if current_time.time() >= self.end_time:
            return True
        
        return None


class TriangleBreakoutStrategy(AdvancedStrategyBase):
    """
    Strategy 3: Triangle Breakout with Confirmation
    
    Requirements:
    - Works on 5m only (not 1m)
    - At least 6 touches
    - Break + retest
    - Volume expansion
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.min_touches = config.get('min_touches', 6)
        self.timeframe = config.get('timeframe', '5min')  # Must be 5min
        
    def detect_triangle(self, df: pd.DataFrame, idx: int, lookback: int = 50) -> Optional[Dict]:
        """Detect triangle pattern."""
        if idx < lookback:
            return None
        
        window = df.iloc[idx-lookback:idx+1]
        
        # Find peaks and troughs
        peaks = []
        troughs = []
        
        for i in range(2, len(window)-2):
            if (window['high'].iloc[i] > window['high'].iloc[i-1] and 
                window['high'].iloc[i] > window['high'].iloc[i+1]):
                peaks.append((i, window['high'].iloc[i]))
            if (window['low'].iloc[i] < window['low'].iloc[i-1] and 
                window['low'].iloc[i] < window['low'].iloc[i+1]):
                troughs.append((i, window['low'].iloc[i]))
        
        if len(peaks) < 3 or len(troughs) < 3:
            return None
        
        # Check if forming triangle (converging trendlines)
        # Simplified: check if recent peaks/troughs are converging
        recent_peaks = sorted(peaks[-3:], key=lambda x: x[1])
        recent_troughs = sorted(troughs[-3:], key=lambda x: x[1])
        
        # Ascending triangle: horizontal resistance, rising support
        # Descending triangle: horizontal support, falling resistance
        # Symmetrical triangle: both converging
        
        total_touches = len(peaks) + len(troughs)
        if total_touches < self.min_touches:
            return None
        
        # Determine triangle type and breakout direction
        if recent_peaks[-1][1] > recent_peaks[0][1] and recent_troughs[-1][1] > recent_troughs[0][1]:
            # Ascending triangle - bullish
            return {'type': 'ascending', 'direction': 'LONG', 'touches': total_touches}
        elif recent_peaks[-1][1] < recent_peaks[0][1] and recent_troughs[-1][1] < recent_troughs[0][1]:
            # Descending triangle - bearish
            return {'type': 'descending', 'direction': 'SHORT', 'touches': total_touches}
        else:
            # Symmetrical - direction depends on breakout
            return {'type': 'symmetrical', 'direction': None, 'touches': total_touches}
    
    def check_entry_conditions(self, row: pd.Series, idx: int, df: pd.DataFrame) -> Optional[str]:
        """Check if entry conditions are met."""
        if self.timeframe != '5min':
            return None  # Only works on 5m
        
        if self.trades_today >= self.max_trades_per_day:
            return None
        if self.current_position is not None:
            return None
        
        # Detect triangle
        triangle = self.detect_triangle(df, idx)
        if triangle is None:
            return None
        
        # Check for breakout
        if idx < 2:
            return None
        
        # Volume expansion
        volume = row.get('volume', 0)
        if idx >= 20:
            avg_volume = df['volume'].iloc[idx-20:idx].mean()
            if volume <= avg_volume * 1.2:  # 20% above average
                return None
        
        # Check breakout direction
        if triangle['direction'] == 'LONG':
            # Breakout above resistance
            if row['close'] > df['high'].iloc[idx-20:idx].max():
                # Check for retest (price came back to breakout level)
                if idx >= 3:
                    recent_low = df['low'].iloc[idx-3:idx].min()
                    resistance = df['high'].iloc[idx-20:idx].max()
                    if recent_low <= resistance * 1.001:  # Retest within 0.1%
                        return 'LONG'
        elif triangle['direction'] == 'SHORT':
            # Breakout below support
            if row['close'] < df['low'].iloc[idx-20:idx].min():
                # Check for retest
                if idx >= 3:
                    recent_high = df['high'].iloc[idx-3:idx].max()
                    support = df['low'].iloc[idx-20:idx].min()
                    if recent_high >= support * 0.999:  # Retest within 0.1%
                        return 'SHORT'
        
        return None
    
    def check_exit_conditions(self, row: pd.Series, vwap: float, current_time: pd.Timestamp) -> bool:
        """Check if exit conditions are met."""
        if self.current_position is None:
            return False
        
        # Stop loss
        if self.current_position == 'LONG':
            if row['low'] <= self.stop_loss:
                return True
        else:
            if row['high'] >= self.stop_loss:
                return True
        
        # End of day
        if current_time.time() >= self.end_time:
            return True
        
        return False
    
    def check_exit_conditions(self, row: pd.Series, vwap: float, current_time: pd.Timestamp) -> bool:
        """Check if exit conditions are met."""
        if self.current_position is None:
            return False
        
        # Stop loss
        if self.current_position == 'LONG':
            if row['low'] <= self.stop_loss:
                return True
            # Exit if VWAP cross back
            if row['close'] < vwap:
                return True
        else:  # SHORT
            if row['high'] >= self.stop_loss:
                return True
            if row['close'] > vwap:
                return True
        
        # End of day
        if current_time.time() >= self.end_time:
            return True
        
        return False


class ThirdCandleORBStrategy(AdvancedStrategyBase):
    """
    Strategy 4: Range Breakout of 3rd 5-min Candle
    
    Requirements:
    - One of the best ORB variants
    - Volume > day avg
    - Trend day filter
    - SL = candle low/high only
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.candle_number = 3  # 3rd candle
        self.timeframe = config.get('timeframe', '5min')
    
    def check_entry_conditions(self, row: pd.Series, idx: int, df: pd.DataFrame) -> Optional[str]:
        """Check if entry conditions are met."""
        if self.trades_today >= self.max_trades_per_day:
            return None
        if self.current_position is not None:
            return None
        
        # Must be on 3rd candle of the day
        current_date = row['date'].date()
        day_data = df[df['date'].dt.date == current_date]
        candle_num = len(day_data[day_data['date'] <= row['date']])
        
        if candle_num != self.candle_number:
            return None
        
        # Calculate range of first 3 candles
        if len(day_data) < 3:
            return None
        
        first_3 = day_data.head(3)
        range_high = first_3['high'].max()
        range_low = first_3['low'].min()
        
        # Volume check
        volume = row.get('volume', 0)
        if len(day_data) >= 20:
            avg_volume = day_data['volume'].head(20).mean()
            if volume <= avg_volume:
                return None
        
        # Trend day filter (simplified: check if price is trending)
        vwap = row.get('vwap', None)
        if pd.isna(vwap):
            return None
        
        close = row['close']
        
        # Breakout above range high
        if close > range_high and close > vwap:
            # Use 3rd candle low as stop loss
            self.stop_loss_candle_low = first_3['low'].iloc[2]
            return 'LONG'
        
        # Breakout below range low
        if close < range_low and close < vwap:
            # Use 3rd candle high as stop loss
            self.stop_loss_candle_high = first_3['high'].iloc[2]
            return 'SHORT'
        
        return None
    
    def calculate_stop_loss(self, entry_price: float, side: str, atr: float) -> float:
        """Override to use candle high/low instead of ATR."""
        if side == 'LONG':
            return getattr(self, 'stop_loss_candle_low', entry_price - (1.5 * atr))
        else:
            return getattr(self, 'stop_loss_candle_high', entry_price + (1.5 * atr))
    
    def check_exit_conditions(self, row: pd.Series, vwap: float, current_time: pd.Timestamp) -> bool:
        """Check if exit conditions are met."""
        if self.current_position is None:
            return False
        
        # Stop loss (candle high/low)
        if self.current_position == 'LONG':
            if row['low'] <= self.stop_loss:
                return True
        else:
            if row['high'] >= self.stop_loss:
                return True
        
        # End of day
        if current_time.time() >= self.end_time:
            return True
        
        return False


class GapOpenStrategy(AdvancedStrategyBase):
    """
    Strategy 8: Gap Open Strategy (2nd 5m candle)
    
    Requirements:
    - Gap > 0.5%
    - Gap in direction of previous trend
    - VWAP confirmation
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.gap_threshold = config.get('gap_threshold', 0.005)  # 0.5%
        self.candle_number = 2  # 2nd candle
    
    def check_entry_conditions(self, row: pd.Series, idx: int, df: pd.DataFrame) -> Optional[str]:
        """Check if entry conditions are met."""
        if self.trades_today >= self.max_trades_per_day:
            return None
        if self.current_position is not None:
            return None
        
        # Must be on 2nd candle of the day
        current_date = row['date'].date()
        day_data = df[df['date'].dt.date == current_date].sort_values('date')
        candle_num = len(day_data[day_data['date'] <= row['date']])
        
        if candle_num != self.candle_number:
            return None
        
        if len(day_data) < 2:
            return None
        
        # Get previous day's close
        prev_day = df[df['date'].dt.date < current_date]
        if len(prev_day) == 0:
            return None
        
        prev_close = prev_day['close'].iloc[-1]
        current_open = day_data['open'].iloc[0]
        
        # Calculate gap
        gap_pct = (current_open - prev_close) / prev_close
        
        # Gap must be > threshold
        if abs(gap_pct) < self.gap_threshold:
            return None
        
        # Check previous trend (last 20 bars of previous day)
        if len(prev_day) >= 20:
            prev_trend = prev_day['close'].iloc[-20:].iloc[-1] - prev_day['close'].iloc[-20:].iloc[0]
            trend_direction = 1 if prev_trend > 0 else -1
        else:
            trend_direction = 1 if gap_pct > 0 else -1
        
        # Gap must be in direction of previous trend
        gap_direction = 1 if gap_pct > 0 else -1
        if gap_direction != trend_direction:
            return None
        
        # VWAP confirmation
        vwap = row.get('vwap', None)
        if pd.isna(vwap):
            return None
        
        close = row['close']
        
        # Gap up with VWAP confirmation
        if gap_pct > 0 and close > vwap:
            return 'LONG'
        
        # Gap down with VWAP confirmation
        if gap_pct < 0 and close < vwap:
            return 'SHORT'
        
        return None
    
    def check_exit_conditions(self, row: pd.Series, vwap: float, current_time: pd.Timestamp) -> bool:
        """Check if exit conditions are met."""
        if self.current_position is None:
            return False
        
        # Stop loss
        if self.current_position == 'LONG':
            if row['low'] <= self.stop_loss:
                return True
            if row['close'] < vwap:
                return True
        else:
            if row['high'] >= self.stop_loss:
                return True
            if row['close'] > vwap:
                return True
        
        # End of day
        if current_time.time() >= self.end_time:
            return True
        
        return False


class BigCandleBreakoutStrategy(AdvancedStrategyBase):
    """
    Strategy 9: Big Candle Breakout Strategy
    
    Requirements:
    - Candle size > 1.5 × ATR(14)
    - Must trade pullback, not candle close
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.candle_size_mult = config.get('candle_size_mult', 1.5)
        self.atr_period = config.get('atr_period', 14)
    
    def check_entry_conditions(self, row: pd.Series, idx: int, df: pd.DataFrame) -> Optional[str]:
        """Check if entry conditions are met."""
        if self.trades_today >= self.max_trades_per_day:
            return None
        if self.current_position is not None:
            return None
        
        if idx < self.atr_period:
            return None
        
        # Check previous candle for big candle
        if idx < 1:
            return None
        
        prev_row = df.iloc[idx-1]
        candle_size = prev_row['high'] - prev_row['low']
        atr = row.get('atr', None)
        
        if pd.isna(atr) or atr == 0:
            return None
        
        # Candle size must be > threshold
        if candle_size < (self.candle_size_mult * atr):
            return None
        
        # Wait for pullback (don't trade on candle close)
        # Entry on pullback to 50% of big candle
        big_candle_mid = (prev_row['high'] + prev_row['low']) / 2
        
        close = row['close']
        high = row['high']
        low = row['low']
        
        # Bullish big candle - wait for pullback
        if prev_row['close'] > prev_row['open']:  # Bullish candle
            # Pullback to mid or lower, then bounce
            if low <= big_candle_mid <= high and close > big_candle_mid:
                return 'LONG'
        
        # Bearish big candle - wait for pullback
        elif prev_row['close'] < prev_row['open']:  # Bearish candle
            # Pullback to mid or higher, then bounce down
            if low <= big_candle_mid <= high and close < big_candle_mid:
                return 'SHORT'
        
        return None
    
    def check_exit_conditions(self, row: pd.Series, vwap: float, current_time: pd.Timestamp) -> bool:
        """Check if exit conditions are met."""
        if self.current_position is None:
            return False
        
        # Stop loss
        if self.current_position == 'LONG':
            if row['low'] <= self.stop_loss:
                return True
        else:
            if row['high'] >= self.stop_loss:
                return True
        
        # End of day
        if current_time.time() >= self.end_time:
            return True
        
        return False


class FlagBreakoutStrategy(AdvancedStrategyBase):
    """
    Strategy 10: Flag Breakout
    
    Requirements:
    - One of the most profitable patterns
    - Only in strong trend days
    - Flag must be ≤ 38% retracement
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.max_retracement = config.get('max_retracement', 0.38)  # 38%
        self.flag_min_bars = config.get('flag_min_bars', 5)
        self.flag_max_bars = config.get('flag_max_bars', 20)
    
    def detect_flag(self, df: pd.DataFrame, idx: int) -> Optional[Dict]:
        """Detect flag pattern."""
        if idx < self.flag_max_bars + 10:
            return None
        
        # Look for strong move (pole)
        window = df.iloc[idx-30:idx+1]
        
        # Find recent high/low
        recent_high_idx = window['high'].idxmax()
        recent_low_idx = window['low'].idxmin()
        
        # Determine trend direction
        if recent_high_idx > recent_low_idx:
            # Uptrend - pole is up move
            pole_start = window.loc[recent_low_idx]
            pole_end = window.loc[recent_high_idx]
            pole_size = pole_end['high'] - pole_start['low']
            direction = 'LONG'
        else:
            # Downtrend - pole is down move
            pole_start = window.loc[recent_high_idx]
            pole_end = window.loc[recent_low_idx]
            pole_size = pole_start['high'] - pole_end['low']
            direction = 'SHORT'
        
        if pole_size == 0:
            return None
        
        # Check for flag (consolidation after pole)
        flag_start_idx = window.index.get_loc(pole_end.name) + 1
        if flag_start_idx >= len(window):
            return None
        
        flag_window = window.iloc[flag_start_idx:]
        
        if len(flag_window) < self.flag_min_bars:
            return None
        
        # Flag must be within max retracement
        if direction == 'LONG':
            flag_high = flag_window['high'].max()
            flag_low = flag_window['low'].min()
            retracement = (pole_end['high'] - flag_low) / pole_size
        else:
            flag_high = flag_window['high'].max()
            flag_low = flag_window['low'].min()
            retracement = (flag_high - pole_end['low']) / pole_size
        
        if retracement > self.max_retracement:
            return None
        
        return {
            'direction': direction,
            'pole_size': pole_size,
            'flag_high': flag_high,
            'flag_low': flag_low,
            'retracement': retracement
        }
    
    def check_entry_conditions(self, row: pd.Series, idx: int, df: pd.DataFrame) -> Optional[str]:
        """Check if entry conditions are met."""
        if self.trades_today >= self.max_trades_per_day:
            return None
        if self.current_position is not None:
            return None
        
        # Detect flag
        flag = self.detect_flag(df, idx)
        if flag is None:
            return None
        
        # Check for breakout from flag
        close = row['close']
        
        if flag['direction'] == 'LONG':
            # Breakout above flag high
            if close > flag['flag_high']:
                # Volume confirmation
                volume = row.get('volume', 0)
                if idx >= 20:
                    avg_volume = df['volume'].iloc[idx-20:idx].mean()
                    if volume > avg_volume * 1.1:  # 10% above average
                        return 'LONG'
        else:  # SHORT
            # Breakout below flag low
            if close < flag['flag_low']:
                volume = row.get('volume', 0)
                if idx >= 20:
                    avg_volume = df['volume'].iloc[idx-20:idx].mean()
                    if volume > avg_volume * 1.1:
                        return 'SHORT'
        
        return None
    
    def check_exit_conditions(self, row: pd.Series, vwap: float, current_time: pd.Timestamp) -> bool:
        """Check if exit conditions are met."""
        if self.current_position is None:
            return False
        
        # Stop loss
        if self.current_position == 'LONG':
            if row['low'] <= self.stop_loss:
                return True
        else:
            if row['high'] >= self.stop_loss:
                return True
        
        # End of day
        if current_time.time() >= self.end_time:
            return True
        
        return False


class CPR20EMAStrategy(AdvancedStrategyBase):
    """
    Strategy 11: CPR + 20 EMA Strategy
    
    Requirements:
    - Excellent day bias tool
    - Works best for range → trend transition days
    - Narrow CPR days
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.cpr_narrow_threshold = config.get('cpr_narrow_threshold', 0.3)  # 30% of ATR
    
    def calculate_cpr(self, prev_day_data: pd.DataFrame) -> Dict:
        """Calculate Central Pivot Range (CPR)."""
        if len(prev_day_data) == 0:
            return None
        
        high = prev_day_data['high'].max()
        low = prev_day_data['low'].min()
        close = prev_day_data['close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        bc = (high + low) / 2  # Bottom Central Pivot
        tc = (pivot - bc) + pivot  # Top Central Pivot
        
        cpr_width = tc - bc
        
        return {
            'pivot': pivot,
            'bc': bc,
            'tc': tc,
            'width': cpr_width
        }
    
    def check_entry_conditions(self, row: pd.Series, idx: int, df: pd.DataFrame) -> Optional[str]:
        """Check if entry conditions are met."""
        if self.trades_today >= self.max_trades_per_day:
            return None
        if self.current_position is not None:
            return None
        
        if idx < 20:
            return None
        
        # Get previous day data
        current_date = row['date'].date()
        prev_day = df[df['date'].dt.date < current_date]
        
        if len(prev_day) == 0:
            return None
        
        # Calculate CPR
        cpr = self.calculate_cpr(prev_day)
        if cpr is None:
            return None
        
        # Check if CPR is narrow
        atr = row.get('atr', None)
        if pd.isna(atr) or atr == 0:
            return None
        
        cpr_width_pct = cpr['width'] / atr
        if cpr_width_pct > self.cpr_narrow_threshold:
            return None  # CPR too wide
        
        # Get 20 EMA
        ema20 = df['ema_20'].iloc[idx] if 'ema_20' in df.columns else None
        if pd.isna(ema20):
            return None
        
        close = row['close']
        
        # Entry logic: Price above CPR + EMA20 = LONG, below = SHORT
        if close > cpr['tc'] and close > ema20:
            return 'LONG'
        elif close < cpr['bc'] and close < ema20:
            return 'SHORT'
        
        return None
    
    def check_exit_conditions(self, row: pd.Series, vwap: float, current_time: pd.Timestamp) -> bool:
        """Check if exit conditions are met."""
        if self.current_position is None:
            return False
        
        # Stop loss
        if self.current_position == 'LONG':
            if row['low'] <= self.stop_loss:
                return True
        else:
            if row['high'] >= self.stop_loss:
                return True
        
        # End of day
        if current_time.time() >= self.end_time:
            return True
        
        return False


class FifteenMinBreakoutStrategy(AdvancedStrategyBase):
    """
    Strategy 12: 15-Minute Breakout Strategy
    
    Requirements:
    - Reliable
    - Fewer trades, better quality
    - Needs VWAP + volume confirmation
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.breakout_period = 15  # 15 minutes
    
    def check_entry_conditions(self, row: pd.Series, idx: int, df: pd.DataFrame) -> Optional[str]:
        """Check if entry conditions are met."""
        if self.trades_today >= self.max_trades_per_day:
            return None
        if self.current_position is not None:
            return None
        
        # Must be at least 15 minutes into the day
        current_date = row['date'].date()
        day_data = df[df['date'].dt.date == current_date].sort_values('date')
        
        if len(day_data) < self.breakout_period:
            return None
        
        # Calculate 15-minute range
        first_15 = day_data.head(self.breakout_period)
        range_high = first_15['high'].max()
        range_low = first_15['low'].min()
        
        # Volume confirmation
        volume = row.get('volume', 0)
        if len(day_data) >= 20:
            avg_volume = day_data['volume'].head(20).mean()
            if volume <= avg_volume * 1.2:
                return None
        
        # VWAP confirmation
        vwap = row.get('vwap', None)
        if pd.isna(vwap):
            return None
        
        close = row['close']
        
        # Breakout above with VWAP confirmation
        if close > range_high and close > vwap:
            return 'LONG'
        
        # Breakout below with VWAP confirmation
        if close < range_low and close < vwap:
            return 'SHORT'
        
        return None
    
    def check_exit_conditions(self, row: pd.Series, vwap: float, current_time: pd.Timestamp) -> bool:
        """Check if exit conditions are met."""
        if self.current_position is None:
            return False
        
        # Stop loss
        if self.current_position == 'LONG':
            if row['low'] <= self.stop_loss:
                return True
            if row['close'] < vwap:
                return True
        else:
            if row['high'] >= self.stop_loss:
                return True
            if row['close'] > vwap:
                return True
        
        # End of day
        if current_time.time() >= self.end_time:
            return True
        
        return False


class Last30MinRangeStrategy(AdvancedStrategyBase):
    """
    Strategy 15: Last 30-min Range of Previous Day
    
    Requirements:
    - Very effective for first-hour trades
    - Needs trend day filter
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.range_minutes = 30
    
    def check_entry_conditions(self, row: pd.Series, idx: int, df: pd.DataFrame) -> Optional[str]:
        """Check if entry conditions are met."""
        if self.trades_today >= self.max_trades_per_day:
            return None
        if self.current_position is not None:
            return None
        
        # Only trade in first hour
        current_time = row['date']
        if current_time.hour > 10 or (current_time.hour == 10 and current_time.minute > 15):
            return None
        
        # Get previous day's last 30 minutes
        current_date = row['date'].date()
        prev_day = df[df['date'].dt.date < current_date]
        
        if len(prev_day) == 0:
            return None
        
        # Get last 30 minutes of previous day
        prev_day_end = prev_day['date'].max()
        range_start = prev_day_end - pd.Timedelta(minutes=self.range_minutes)
        last_30min = prev_day[prev_day['date'] >= range_start]
        
        if len(last_30min) == 0:
            return None
        
        range_high = last_30min['high'].max()
        range_low = last_30min['low'].min()
        
        # Trend day filter: check if previous day was trending
        prev_day_high = prev_day['high'].max()
        prev_day_low = prev_day['low'].min()
        prev_day_close = prev_day['close'].iloc[-1]
        prev_day_open = prev_day['open'].iloc[0]
        
        # Strong trend if close is near high or low
        trend_strength = abs(prev_day_close - prev_day_open) / (prev_day_high - prev_day_low)
        if trend_strength < 0.5:  # Not a strong trend day
            return None
        
        close = row['close']
        
        # Breakout above range
        if close > range_high:
            return 'LONG'
        
        # Breakout below range
        if close < range_low:
            return 'SHORT'
        
        return None
    
    def check_exit_conditions(self, row: pd.Series, vwap: float, current_time: pd.Timestamp) -> bool:
        """Check if exit conditions are met."""
        if self.current_position is None:
            return False
        
        # Stop loss
        if self.current_position == 'LONG':
            if row['low'] <= self.stop_loss:
                return True
        else:
            if row['high'] >= self.stop_loss:
                return True
        
        # Exit after first hour
        if current_time.hour > 10 or (current_time.hour == 10 and current_time.minute > 15):
            return True
        
        # End of day
        if current_time.time() >= self.end_time:
            return True
        
        return False


class EMA20MeanReversionStrategy(AdvancedStrategyBase):
    """
    Strategy 17: 20 EMA Mean Reversion
    
    Requirements:
    - Works ONLY in range days
    - Must be disabled on trend days
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.lookback = 20
    
    def is_range_day(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if current day is a range day."""
        current_date = df.iloc[idx]['date'].date()
        day_data = df[df['date'].dt.date == current_date]
        
        if len(day_data) < 10:
            return False
        
        # Range day: price oscillates, doesn't trend strongly
        day_high = day_data['high'].max()
        day_low = day_data['low'].min()
        day_range = day_high - day_low
        
        # Check if price is oscillating (multiple touches of high/low)
        high_touches = (day_data['high'] >= day_high * 0.998).sum()
        low_touches = (day_data['low'] <= day_low * 1.002).sum()
        
        # Range day if multiple touches and not strong directional move
        if high_touches >= 2 and low_touches >= 2:
            # Check if close is in middle of range (not trending)
            day_close = day_data['close'].iloc[-1]
            range_mid = (day_high + day_low) / 2
            if abs(day_close - range_mid) / day_range < 0.3:  # Close near middle
                return True
        
        return False
    
    def check_entry_conditions(self, row: pd.Series, idx: int, df: pd.DataFrame) -> Optional[str]:
        """Check if entry conditions are met."""
        if self.trades_today >= self.max_trades_per_day:
            return None
        if self.current_position is not None:
            return None
        
        if idx < self.lookback:
            return None
        
        # Must be a range day
        if not self.is_range_day(df, idx):
            return None
        
        # Get 20 EMA
        ema20 = df['ema_20'].iloc[idx] if 'ema_20' in df.columns else None
        if pd.isna(ema20):
            return None
        
        close = row['close']
        high = row['high']
        low = row['low']
        
        # Mean reversion: buy when price dips below EMA20, sell when above
        # Price pulled back to EMA20
        if low <= ema20 <= high:
            # If close is below EMA20, expect bounce up (LONG)
            if close < ema20:
                # Check RSI for oversold
                rsi = df['rsi'].iloc[idx] if 'rsi' in df.columns else 50
                if rsi < 40:  # Oversold
                    return 'LONG'
            # If close is above EMA20, expect pullback down (SHORT)
            elif close > ema20:
                # Check RSI for overbought
                rsi = df['rsi'].iloc[idx] if 'rsi' in df.columns else 50
                if rsi > 60:  # Overbought
                    return 'SHORT'
        
        return None
    
    def check_exit_conditions(self, row: pd.Series, vwap: float, current_time: pd.Timestamp) -> bool:
        """Check if exit conditions are met."""
        if self.current_position is None:
            return False
        
        # Stop loss
        if self.current_position == 'LONG':
            if row['low'] <= self.stop_loss:
                return True
            # Exit when price returns to EMA20
            ema20 = row.get('ema_20', None)
            if not pd.isna(ema20) and row['close'] >= ema20:
                return True
        else:
            if row['high'] >= self.stop_loss:
                return True
            # Exit when price returns to EMA20
            ema20 = row.get('ema_20', None)
            if not pd.isna(ema20) and row['close'] <= ema20:
                return True
        
        # End of day
        if current_time.time() >= self.end_time:
            return True
        
        return False


# Strategy factory
def get_strategy(strategy_name: str, config: Dict) -> AdvancedStrategyBase:
    """Get strategy instance by name."""
    strategies = {
        'holy_grail': HolyGrailStrategy,
        'triangle_breakout': TriangleBreakoutStrategy,
        'third_candle_orb': ThirdCandleORBStrategy,
        'gap_open': GapOpenStrategy,
        'big_candle_breakout': BigCandleBreakoutStrategy,
        'flag_breakout': FlagBreakoutStrategy,
        'cpr_20ema': CPR20EMAStrategy,
        'fifteen_min_breakout': FifteenMinBreakoutStrategy,
        'last_30min_range': Last30MinRangeStrategy,
        'ema20_mean_reversion': EMA20MeanReversionStrategy,
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_name](config)

