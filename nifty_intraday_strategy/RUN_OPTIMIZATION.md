# How to Run Strategy Optimization

## Quick Start

### Step 1: Navigate to Strategy Directory

```bash
cd /Users/ankurkumar/Documents/CODE/Trading_Nifty_50/nifty_intraday_strategy
```

### Step 2: Run Optimization Script

```bash
python3 src/optimize_strategy.py
```

Or if `python` command works:

```bash
python src/optimize_strategy.py
```

## What Happens Next

1. **The script will load your data** (`data/nifty_1min.csv`)
2. **You'll be asked to choose optimization method**:
   - Enter `1` for Grid Search (comprehensive but slow - tests all combinations)
   - Enter `2` for Random Search (faster, recommended - tests random combinations)
   - Enter `3` for Both (runs both methods)

3. **If you choose Random Search**, you'll be asked:
   - "Enter number of random iterations (default 100):"
   - Press Enter for default (100) or type a number (e.g., 50 for faster, 200 for more thorough)

4. **The optimization will run**:
   - Progress will be shown every 10 iterations
   - Best results so far will be displayed
   - This may take 30 minutes to several hours depending on:
     - Number of iterations
     - Size of your data file
     - Your computer's speed

5. **Results will be displayed**:
   - Top 30 parameter combinations
   - Summary statistics
   - Best parameters will be saved to `config_optimized.yaml`

## Example Session

```
$ python3 src/optimize_strategy.py

Loading configuration...
Loading data from data/nifty_1min.csv...
Loaded 975323 bars
Date range: 2015-01-01 09:15:00 to 2025-07-25 15:30:00

======================================================================
STRATEGY OPTIMIZATION
======================================================================

Choose optimization method:
1. Grid Search (comprehensive but slow)
2. Random Search (faster, recommended)
3. Both

Enter choice (1/2/3): 2

Enter number of random iterations (default 100): 50

======================================================================
RANDOM SEARCH OPTIMIZATION
======================================================================

Testing 50 random parameter combinations...

Progress: 10/50 (20.0%)
  New best: PnL=₹1234.56, WinRate=45.2%, Trades=42
Progress: 20/50 (40.0%)
  New best: PnL=₹5678.90, WinRate=52.1%, Trades=38
...

======================================================================
TOP 30 PARAMETER COMBINATIONS
======================================================================

[Results table will be displayed here]

======================================================================
SUMMARY STATISTICS
======================================================================

Total combinations tested: 50
Profitable combinations: 12
Best total PnL: ₹5678.90
Best win rate: 52.1%
Best profit factor: 1.85
Best Sharpe ratio: 1.23

Optimization results saved to: results/optimization_results_20231217_230500.csv
Best parameters saved to: config_optimized.yaml

======================================================================
BEST PARAMETERS FOUND
======================================================================
Opening Range Period: 20
ATR Multiplier: 2.0
Risk Per Trade: 0.01
RSI Oversold: 30
RSI Overbought: 70
Min Score: 3

Expected Performance:
  Total PnL: ₹5678.90
  Win Rate: 52.1%
  Profit Factor: 1.85
  Sharpe Ratio: 1.23
```

## Recommended Settings

### For Quick Test (10-15 minutes)
- Choose: **Random Search**
- Iterations: **20-30**

### For Thorough Optimization (1-2 hours)
- Choose: **Random Search**
- Iterations: **100-200**

### For Most Comprehensive (Several hours)
- Choose: **Grid Search** or **Both**
- This will test all combinations (thousands of tests)

## After Optimization

1. **Review the best parameters** in `config_optimized.yaml`
2. **Check the results CSV** in `results/optimization_results_*.csv`
3. **Run a backtest** with optimized parameters:
   ```bash
   python3 src/run_enhanced_backtest.py
   ```
   (But first, copy optimized parameters to `config.yaml` or use the optimized config)

## Troubleshooting

### Error: "Data file not found"
- Make sure `data/nifty_1min.csv` exists
- Check you're in the correct directory

### Error: "Module not found"
- Install dependencies: `pip3 install -r requirements.txt`

### Optimization takes too long
- Use Random Search instead of Grid Search
- Reduce number of iterations (20-50)
- Disable slow features in `config.yaml`:
  - Set `use_chart_patterns: false` (already disabled)
  - Set `use_multiple_timeframe: false` temporarily

### No profitable results
- Try increasing iterations (200+)
- Adjust parameter ranges in `optimize_strategy.py`
- Check if your data has sufficient trading activity

## Tips

1. **Start with Random Search (50 iterations)** to get quick results
2. **If you find promising parameters**, run Random Search again with more iterations (200+) focusing on those parameter ranges
3. **Save your results** - the CSV file contains all tested combinations
4. **Validate on different time periods** - test optimized parameters on data not used for optimization

## Files Created

- `results/optimization_results_TIMESTAMP.csv` - All optimization results
- `config_optimized.yaml` - Best parameters found
- Console output shows top results and summary

## Next Steps

After finding good parameters:
1. Review the best parameters
2. Run backtest with optimized config
3. Validate on out-of-sample data if available
4. Consider paper trading before live trading

