# Nifty Intraday Trading Strategy Backtester

A comprehensive Python project for backtesting an advanced intraday Nifty trading strategy using 1-minute OHLCV data with multiple technical indicators, patterns, and analysis tools.

## Strategy Overview

This backtester implements an **Enhanced Opening Range Breakout (ORB) strategy** that combines:

### Base Strategy
- **Opening Range**: First 15 minutes after market open (9:15 AM)
- **Entry Conditions**:
  - **LONG**: Price closes above OR high AND close > VWAP
  - **SHORT**: Price closes below OR low AND close < VWAP
- **Stop Loss**: ATR-based (entry ± 1.5 × ATR)
- **Exit Conditions**:
  - Stop loss hit
  - VWAP cross back
  - End of day (3:15 PM)
- **Execution**: Realistic execution with slippage (0.015%) and commission (0.03%)

### Enhanced Features

#### 1. **Candlestick Patterns**
- Marubozu (Bullish/Bearish)
- Hammer (Bullish reversal)
- Hanging Man (Bearish reversal)
- Shooting Star (Bearish reversal)
- Doji (Indecision)
- Engulfing (Bullish/Bearish)
- Piercing Pattern (Bullish)
- Dark Cloud Cover (Bearish)

#### 2. **Technical Indicators**
- **Moving Averages**: SMA (20, 50), EMA (12, 26)
- **RSI**: Relative Strength Index (14 period)
- **MACD**: Moving Average Convergence Divergence
- **Chaikin Oscillator**: Accumulation/Distribution momentum
- **Williams Fractals**: Reversal point identification
- **Bollinger Bands**: Volatility bands
- **Stochastic Oscillator**: Momentum indicator

#### 3. **Chart Patterns**
- Head and Shoulders / Inverse Head and Shoulders
- Double Top / Double Bottom
- Triple Top / Triple Bottom
- N-tops / N-bottoms (4+ peaks/troughs)

#### 4. **Price Action Tools**
- **Pivot Points**: Standard, Fibonacci, and Camarilla methods
- **Fibonacci Retracement**: Support/resistance levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
- **Support/Resistance Identification**: Automatic level detection

#### 5. **Volatility Measures**
- **ATR**: Average True Range
- **Standard Deviation**: Volatility of returns
- **Parkinson Volatility**: High-low range estimator
- **Garman-Klass Volatility**: OHLC-based estimator
- **Volatility Ratio**: Short-term vs long-term volatility

#### 6. **Multiple Timeframe Analysis**
- **15-minute**: Short-term trend
- **1-hour**: Medium-term trend
- **1-day**: Long-term trend
- **Weekly**: Very long-term trend (optional)
- Cross-timeframe trend alignment
- Multi-timeframe support/resistance
- Multi-timeframe momentum indicators

## Project Structure

```
nifty_intraday_strategy/
├── data/
│   └── nifty_1min.csv              # Your 1-minute OHLCV data
├── src/
│   ├── indicators.py               # VWAP, ATR calculations
│   ├── strategy.py                 # Base ORB strategy
│   ├── enhanced_strategy.py        # Enhanced strategy with all indicators
│   ├── candlestick_patterns.py     # Candlestick pattern recognition
│   ├── advanced_indicators.py      # RSI, MACD, Chaikin, etc.
│   ├── chart_patterns.py           # Chart pattern detection
│   ├── price_action.py              # Pivot points, Fibonacci
│   ├── volatility.py                # Volatility measures
│   ├── multi_timeframe.py           # Multiple timeframe analysis
│   ├── backtester.py                # Base execution engine
│   ├── enhanced_backtester.py       # Enhanced backtester
│   ├── utils.py                     # Date filters, plotting utils
│   ├── run_backtest.py              # Basic backtest script
│   └── run_enhanced_backtest.py     # Enhanced backtest script
├── results/                         # Generated after backtest
│   ├── trades.csv / enhanced_trades.csv
│   ├── equity_curve.csv / enhanced_equity_curve.csv
│   └── metrics.json / enhanced_metrics.json
├── config.yaml                      # Strategy & backtest parameters
├── requirements.txt
└── README.md
```

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd nifty_intraday_strategy
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

1. **Place your data file**:
   - Copy your `NIFTY 50_minute.csv` file to the `data/` directory
   - Rename it to `nifty_1min.csv`
   - The CSV should have columns: `date,open,high,low,close,volume`

2. **Configure parameters** (optional):
   - Edit `config.yaml` to adjust strategy parameters, capital, costs, etc.

## Running the Backtest

### Basic Backtest (ORB + VWAP + ATR only)
```bash
python src/run_backtest.py
```

### Enhanced Backtest (All indicators and patterns)
```bash
python src/run_enhanced_backtest.py
```

**Note**: The enhanced backtest takes longer to run as it calculates all indicators and patterns. Progress is shown during execution.

## Output

The backtest will generate:

1. **Console Output**:
   - Backtest summary with key metrics
   - Trade list (first 10 trades)

2. **Plots**:
   - Equity curve
   - PnL histogram

3. **CSV Files** (in `results/` directory):
   - `trades.csv` or `enhanced_trades.csv`: Complete list of all trades
   - `equity_curve.csv` or `enhanced_equity_curve.csv`: Daily equity values
   - `metrics.json` or `enhanced_metrics.json`: Performance metrics in JSON format

## Performance Metrics

The backtest calculates:

- **Capital & Returns**: Initial capital, final equity, total return, total PnL
- **Trade Statistics**: Total trades, winning/losing trades, win rate
- **PnL Metrics**: Average PnL per trade, average win/loss, profit factor
- **Risk Metrics**: Maximum drawdown, Sharpe ratio

## Configuration Parameters

Edit `config.yaml` to customize:

### Basic Parameters
- `initial_capital`: Starting capital (default: ₹1,000)
- `risk_per_trade`: Risk per trade as fraction of equity (default: 0.5%)
- `n_or`: Opening range period in minutes (default: 15)
- `atr_period`: ATR calculation period (default: 14)
- `atr_mult`: ATR multiplier for stop loss (default: 1.5)
- `start_time`: Market open time (default: "09:15")
- `end_time`: Market close time (default: "15:15")
- `max_trades_per_day`: Maximum trades per day (default: 1)
- `slippage`: Slippage as fraction (default: 0.015%)
- `commission`: Commission per side as fraction (default: 0.03%)

### Enhanced Strategy Features
- `use_candlestick`: Enable candlestick patterns (default: true)
- `use_indicators`: Enable technical indicators (default: true)
- `use_chart_patterns`: Enable chart patterns (default: true)
- `use_price_action`: Enable price action tools (default: true)
- `use_volatility`: Enable volatility measures (default: true)
- `use_multiple_timeframe`: Enable multiple timeframe analysis (default: true)

### Indicator Parameters
- `rsi_oversold`: RSI oversold level (default: 30)
- `rsi_overbought`: RSI overbought level (default: 70)
- `macd_threshold`: MACD threshold (default: 0)
- `mtf_timeframes`: List of timeframes for multi-timeframe analysis (default: ["15min", "1H", "1D"])

### Pattern Parameters
- `pattern_tolerance`: Price tolerance for pattern matching (default: 0.02 = 2%)
- `pattern_min_distance`: Minimum distance between pattern points (default: 10 bars)

### Price Action Parameters
- `fib_period`: Period for Fibonacci retracement (default: 20)
- `sr_window`: Window for support/resistance (default: 20)
- `sr_tolerance`: Tolerance for support/resistance levels (default: 0.01 = 1%)

### Volatility Parameters
- `volatility_period`: Period for volatility calculation (default: 20)
- `vol_short`: Short period for volatility ratio (default: 10)
- `vol_long`: Long period for volatility ratio (default: 20)

## Strategy Logic

The enhanced strategy uses a **scoring system** to determine entry signals:

1. **Base ORB Condition**: Must be met (price breaks OR high/low with VWAP confirmation)
2. **Indicator Scoring**: Each indicator/pattern adds points:
   - **Bullish Signals**: +1 to +2 points
   - **Bearish Signals**: +1 to +2 points
3. **Volatility Filter**: Reduces scores in high volatility regimes
4. **Multi-Timeframe Alignment**: Higher weight for aligned trends
5. **Entry Decision**: Requires minimum score (default: 2) and higher score for chosen direction

This ensures only high-probability trades are taken, filtering out noise and false signals.

## Features

✅ **No Lookahead Bias**: Uses only past data for decisions  
✅ **Realistic Execution**: Next-minute fills with slippage and commission  
✅ **Day-by-Day Processing**: Proper handling of daily resets  
✅ **Comprehensive Metrics**: Full performance analysis  
✅ **Visualization**: Equity curve and PnL distribution plots  
✅ **Modular Design**: Clean, maintainable code structure  
✅ **Advanced Indicators**: 20+ technical indicators  
✅ **Pattern Recognition**: Automated candlestick and chart pattern detection  
✅ **Multi-Timeframe**: Analysis across 15min, 1H, 1D timeframes  
✅ **Volatility Analysis**: Multiple volatility measures and regimes  

## Notes

- The strategy only takes 1 trade per day (configurable)
- All positions are closed at end of day (3:15 PM)
- Stop loss is calculated using ATR and applied immediately
- VWAP cross-back exits when price crosses VWAP in opposite direction
- Position sizing is based on risk per trade and stop loss distance
- Enhanced backtest may take 10-30 minutes for 10 years of 1-minute data
- All indicators are calculated without lookahead bias

## Troubleshooting

**Error: Data file not found**
- Make sure `nifty_1min.csv` is in the `data/` directory
- Check that the file name matches exactly

**Error: Module not found**
- Run `pip install -r requirements.txt` to install dependencies
- Make sure you're running from the project root directory

**No trades generated**
- Check that your data has sufficient volume
- Verify that entry conditions are being met
- Adjust strategy parameters in `config.yaml` if needed
- Try reducing the minimum score threshold in the enhanced strategy

**Memory issues with large datasets**
- The enhanced backtest loads all indicators into memory
- For very large datasets (>1M bars), consider processing in chunks
- Reduce the number of enabled features if needed

## Performance Tips

1. **Start with basic backtest** to verify data and setup
2. **Enable features gradually** to see their impact
3. **Adjust scoring thresholds** in `enhanced_strategy.py` to filter trades
4. **Use multiple timeframes** for better trend confirmation
5. **Monitor volatility regimes** to avoid trading in extreme conditions

## License

This project is provided as-is for educational and research purposes.
