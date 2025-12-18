# Advanced Strategies for NIFTY Backtesting

This document describes the high-potential strategies implemented for NIFTY 1m/5m timeframes.

## How to Run

### Quick Start

```bash
cd /Users/ankurkumar/Documents/CODE/Trading_Nifty_50/nifty_intraday_strategy
python3 src/run_advanced_strategies.py
```

Then select a strategy number (1-10) from the menu.

### Available Strategies

1. **Holy Grail Strategy** - EMA pullback in strong trend
2. **Triangle Breakout** - With confirmation (5m only)
3. **3rd Candle ORB** - Range breakout of 3rd 5-min candle
4. **Gap Open Strategy** - 2nd 5m candle
5. **Big Candle Breakout** - Trade pullback, not candle close
6. **Flag Breakout** - Most profitable pattern
7. **CPR + 20 EMA** - Day bias tool
8. **15-Minute Breakout** - Reliable, fewer trades
9. **Last 30-min Range** - Previous day's range
10. **20 EMA Mean Reversion** - Range days only

## Strategy Details

### 1. Holy Grail Strategy

**Requirements:**
- 20 EMA > 50 EMA (uptrend) or 20 EMA < 50 EMA (downtrend)
- ADX > 20 (strong trend)
- VWAP in trend direction
- ATR expanding
- Entry on EMA pullback

**Key Points:**
- Works ONLY IF all conditions are met
- Without ADX & VWAP → fails badly
- Requires strong trend environment

### 2. Triangle Breakout with Confirmation

**Requirements:**
- Works on 5m only (not 1m)
- At least 6 touches
- Break + retest
- Volume expansion

**Key Points:**
- 1m timeframe = noise → reject
- Must wait for retest after breakout
- Volume must be 20% above average

### 3. Range Breakout of 3rd 5-min Candle

**Requirements:**
- One of the best ORB variants
- Volume > day avg
- Trend day filter
- SL = candle low/high only (not ATR)

**Key Points:**
- Uses 3rd candle's high/low as stop loss
- Requires VWAP confirmation
- Only trades on trend days

### 4. Gap Open Strategy (2nd 5m candle)

**Requirements:**
- Gap > 0.5%
- Gap in direction of previous trend
- VWAP confirmation

**Key Points:**
- Very strong on NIFTY
- Only trades 2nd candle of the day
- Gap must align with previous day's trend

### 5. Big Candle Breakout Strategy

**Requirements:**
- Candle size > 1.5 × ATR(14)
- Must trade pullback, not candle close

**Key Points:**
- Works well
- Waits for pullback to 50% of big candle
- Entry on bounce from pullback

### 6. Flag Breakout

**Requirements:**
- One of the most profitable patterns
- Only in strong trend days
- Flag must be ≤ 38% retracement

**Key Points:**
- Requires strong pole (trend move)
- Flag consolidation must be shallow
- Volume expansion on breakout

### 7. CPR + 20 EMA Strategy

**Requirements:**
- Excellent day bias tool
- Works best for range → trend transition days
- Narrow CPR days

**Key Points:**
- CPR width must be < 30% of ATR
- Price above CPR + EMA20 = LONG
- Price below CPR + EMA20 = SHORT

### 8. 15-Minute Breakout Strategy

**Requirements:**
- Reliable
- Fewer trades, better quality
- Needs VWAP + volume confirmation

**Key Points:**
- Uses first 15 minutes as range
- Breakout must have volume > 20% above average
- VWAP must confirm direction

### 9. Last 30-min Range of Previous Day

**Requirements:**
- Very effective for first-hour trades
- Needs trend day filter

**Key Points:**
- Only trades in first hour (9:15-10:15)
- Previous day must be strong trend day
- Breakout of last 30 minutes' range

### 10. 20 EMA Mean Reversion

**Requirements:**
- Works ONLY in range days
- Must be disabled on trend days

**Key Points:**
- Detects range days automatically
- Buys dips below EMA20 (oversold)
- Sells rallies above EMA20 (overbought)
- Exits when price returns to EMA20

## Output Files

Each strategy generates:
- `{strategy_id}_trades.csv` - All trades
- `{strategy_id}_equity_curve.csv` - Equity curve
- `{strategy_id}_metrics.json` - Performance metrics

## Configuration

Edit `config.yaml` to adjust:
- `atr_mult`: Stop loss multiplier (default: 1.5)
- `risk_per_trade`: Risk per trade (default: 0.5%)
- `max_trades_per_day`: Maximum trades per day (default: 1)

Strategy-specific parameters can be added to config:
- `adx_threshold`: For Holy Grail (default: 20)
- `gap_threshold`: For Gap Open (default: 0.005 = 0.5%)
- `candle_size_mult`: For Big Candle (default: 1.5)
- `max_retracement`: For Flag (default: 0.38 = 38%)
- `cpr_narrow_threshold`: For CPR (default: 0.3 = 30%)

## Notes

- All strategies require proper indicator calculation
- Some strategies work better on 5m vs 1m (noted in descriptions)
- Strategies automatically filter for appropriate market conditions
- End-of-day exits are enforced for all strategies
- Stop losses are ATR-based unless specified otherwise

## Next Steps

1. Run each strategy individually to see performance
2. Compare results across strategies
3. Optimize parameters for best-performing strategies
4. Consider combining multiple strategies

