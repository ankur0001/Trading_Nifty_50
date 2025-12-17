# Timeframe Dropdown Feature

## Overview
Added a timeframe dropdown in the header (left side) that allows users to switch between different candle timeframes. The candles are automatically consolidated/resampled from the 1-minute data to the selected timeframe.

## Available Timeframes
- **1m** - 1 minute (original data)
- **2m** - 2 minutes
- **3m** - 3 minutes
- **5m** - 5 minutes
- **15m** - 15 minutes
- **30m** - 30 minutes
- **45m** - 45 minutes

## How It Works

### Candle Consolidation
When switching timeframes, the system:
1. Takes the raw 1-minute candle data
2. Groups candles into buckets based on the selected timeframe
3. Creates consolidated candles with:
   - **Open**: First candle's open price in the bucket
   - **High**: Maximum high price in the bucket
   - **Low**: Minimum low price in the bucket
   - **Close**: Last candle's close price in the bucket
   - **Volume**: Sum of all volumes in the bucket

### Example: 5-minute Consolidation
If you have 1-minute candles:
- 9:15, 9:16, 9:17, 9:18, 9:19 → One 5m candle (9:15-9:19)
- 9:20, 9:21, 9:22, 9:23, 9:24 → One 5m candle (9:20-9:24)

## Features

### 1. **Visual Feedback**
   - Active timeframe is highlighted in blue
   - Button shows current timeframe (e.g., "5m ▾")
   - Dropdown shows all available options

### 2. **Smooth Transitions**
   - Chart view is maintained when switching (zoomed area stays visible)
   - Indicators recalculate automatically
   - No data loss or disruption

### 3. **Replay Integration**
   - If replay is active, it stops when timeframe changes
   - Replay works with any selected timeframe
   - Data is resampled before replay starts

### 4. **Data Loading**
   - Works with all data loading methods:
     - Latest data
     - Range selection
     - Before/After loading
   - Always respects the current timeframe setting

## UI Location
The timeframe dropdown is located in the **header-left** section, before the "Indicators" dropdown.

## Usage
1. Click on the timeframe button (e.g., "1m ▾")
2. Select desired timeframe from the dropdown
3. Chart automatically updates with consolidated candles
4. All indicators recalculate for the new timeframe

## Technical Details

### Resampling Algorithm
The `resampleCandles()` function:
- Groups 1-minute candles into time buckets
- Each bucket represents one candle in the target timeframe
- Buckets are aligned to time boundaries (e.g., 5m buckets start at :00, :05, :10, etc.)

### Indicator Recalculation
When timeframe changes:
- All indicators (SMA, EMA, RSI, MACD, etc.) are recalculated
- Uses the consolidated candle data
- Maintains indicator accuracy

### State Management
- Current timeframe stored in `currentTF` variable
- Default is 1 minute
- Persists across data loads until changed

## Notes
- Switching timeframes does not reload data from server
- All resampling happens client-side from 1-minute data
- Faster timeframes (1m, 2m, 3m) show more detail
- Slower timeframes (15m, 30m, 45m) show broader trends
- Best practice: Use 1m-5m for intraday trading, 15m-45m for swing analysis

