# Strategy Optimization Guide

## Overview

This guide explains how to optimize the Nifty intraday trading strategy to maximize profit using the optimization framework.

## Current Performance Issues

Based on the latest backtest results:
- **Win Rate**: 7.69% (only 1 winning trade out of 13)
- **Total PnL**: -₹16,255.48 (huge loss)
- **Profit Factor**: 0.000013 (extremely low)
- **Max Drawdown**: -95.37%

The strategy is clearly unprofitable and needs optimization.

## Optimization Framework

A comprehensive optimization script (`src/optimize_strategy.py`) has been created to test different parameter combinations and find the best settings.

### Features

1. **Grid Search**: Tests all combinations of specified parameter ranges (comprehensive but slow)
2. **Random Search**: Tests random parameter combinations (faster, recommended)
3. **Parameter Testing**:
   - Opening Range Period (n_or): 10-30 minutes
   - ATR Multiplier (atr_mult): 1.0-3.5
   - Risk Per Trade: 0.1%-3%
   - RSI Oversold: 20-45
   - RSI Overbought: 55-85
   - Minimum Score: 1-5

## Running Optimization

### Step 1: Navigate to Strategy Directory

```bash
cd nifty_intraday_strategy
```

### Step 2: Run Optimization Script

```bash
python src/optimize_strategy.py
```

### Step 3: Choose Optimization Method

The script will prompt you to choose:
1. **Grid Search** - Comprehensive but slow (tests all combinations)
2. **Random Search** - Faster, recommended (tests random combinations)
3. **Both** - Runs both methods

For initial optimization, **Random Search with 50-100 iterations** is recommended.

### Step 4: Review Results

The script will:
- Display top 30 parameter combinations
- Show summary statistics
- Save results to `results/optimization_results_TIMESTAMP.csv`
- Save best parameters to `config_optimized.yaml`

## Understanding Results

### Key Metrics to Look For

1. **Total PnL**: Should be positive and as high as possible
2. **Win Rate**: Should be > 50% ideally, but > 40% can work with good risk management
3. **Profit Factor**: Should be > 1.5 (gross profit / gross loss)
4. **Sharpe Ratio**: Should be > 1.0 (risk-adjusted returns)
5. **Max Drawdown**: Should be < 30% (lower is better)

### Example Good Result

```
Total PnL: ₹50,000+
Win Rate: 45%+
Profit Factor: 1.5+
Sharpe Ratio: 1.2+
Max Drawdown: < 25%
```

## Using Optimized Parameters

After optimization:

1. **Review Best Parameters**: Check `config_optimized.yaml`
2. **Test on Out-of-Sample Data**: Run backtest with optimized parameters on different time period
3. **Fine-Tune**: Adjust parameters slightly based on results
4. **Update Config**: Copy best parameters to `config.yaml` if satisfied

## Parameter Tuning Tips

### Opening Range Period (n_or)
- **Shorter (10-15 min)**: More trades, more false signals
- **Longer (20-30 min)**: Fewer trades, stronger signals
- **Recommendation**: Start with 15-20 minutes

### ATR Multiplier (atr_mult)
- **Lower (1.0-1.5)**: Tighter stops, more stop-outs
- **Higher (2.5-3.5)**: Wider stops, fewer stop-outs but larger losses
- **Recommendation**: Start with 1.5-2.0

### Risk Per Trade
- **Lower (0.2%-0.5%)**: Conservative, slower growth
- **Higher (1%-2%)**: Aggressive, faster growth but higher risk
- **Recommendation**: Start with 0.5%-1%

### RSI Levels
- **Oversold (20-30)**: More signals, more false signals
- **Oversold (35-45)**: Fewer signals, stronger signals
- **Overbought (60-70)**: More signals
- **Overbought (75-85)**: Fewer signals, stronger signals
- **Recommendation**: Start with 30/70

### Minimum Score
- **Lower (1-2)**: More trades, more false signals
- **Higher (3-5)**: Fewer trades, stronger signals
- **Recommendation**: Start with 2-3

## Common Issues and Solutions

### Issue: All Results Show Negative PnL

**Solutions**:
1. Increase minimum score threshold (filter out weak signals)
2. Widen stop loss (atr_mult = 2.0-2.5)
3. Reduce risk per trade (0.2%-0.5%)
4. Test different opening range periods

### Issue: Too Few Trades

**Solutions**:
1. Decrease minimum score threshold
2. Adjust RSI levels (wider range)
3. Reduce opening range period

### Issue: Too Many Losing Trades

**Solutions**:
1. Increase minimum score threshold
2. Add more filters (enable chart patterns, multi-timeframe)
3. Tighten entry conditions
4. Test different timeframes

## Next Steps After Optimization

1. **Walk-Forward Analysis**: Test optimized parameters on different time periods
2. **Monte Carlo Simulation**: Test robustness of strategy
3. **Paper Trading**: Test in live market with paper trading
4. **Live Trading**: Start with small position sizes

## Files Created

- `results/optimization_results_TIMESTAMP.csv`: All optimization results
- `config_optimized.yaml`: Best parameters found
- `results/enhanced_metrics.json`: Latest backtest metrics

## Notes

- Optimization can take 30 minutes to several hours depending on data size and method chosen
- Use caching (enabled by default) to speed up reruns
- Always validate optimized parameters on out-of-sample data
- Consider transaction costs and slippage in real trading

## Support

If you encounter issues:
1. Check that data file exists: `data/nifty_1min.csv`
2. Verify config file: `config.yaml`
3. Check Python dependencies: `pip install -r requirements.txt`
4. Review error messages in console output

