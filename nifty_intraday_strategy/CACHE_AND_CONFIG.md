# Strategy Caching and Configuration Guide

## Overview

The strategy backtesting system now includes:
1. **Caching** - Results are cached to avoid re-running expensive calculations
2. **Strategy Configuration** - Enable/disable strategies via config file

## Strategy Configuration File

Edit `strategies_config.yaml` to enable/disable strategies:

```yaml
strategies:
  holy_grail:
    enabled: true  # Set to false to skip this strategy
  
  triangle_breakout:
    enabled: false  # Disabled - takes too long (34+ hours)
  
  # ... other strategies
```

### Default Configuration

By default, `triangle_breakout` is **disabled** because it takes 34+ hours to run. All other strategies are enabled.

## Caching System

### How It Works

1. **Cache Key Generation**: 
   - Based on strategy name
   - Data hash (first/last dates, length)
   - Config parameters (ATR, risk, capital, etc.)

2. **Cache Storage**:
   - Location: `cache/strategy_results/`
   - Format: JSON files named `{strategy}_{data_hash}_{config_hash}.json`

3. **Cache Expiry**:
   - Default: 7 days
   - Configurable in `strategies_config.yaml`
   - Set to 0 for no expiry

### Cache Configuration

In `strategies_config.yaml`:

```yaml
cache:
  enabled: true  # Enable/disable caching
  cache_dir: "cache/strategy_results"  # Cache directory
  cache_expiry_days: 7  # Days before cache expires (0 = no expiry)
```

### When Cache is Used

Cache is used when:
- Same strategy name
- Same data file (same dates, same length)
- Same configuration parameters

Cache is **NOT** used when:
- Data file changes
- Configuration parameters change
- Cache is disabled
- Cache has expired

### Cache Benefits

- **Speed**: Cached results load in seconds vs hours
- **Cost Savings**: Avoid re-running expensive calculations
- **Consistency**: Same results for same inputs

## Usage Examples

### Disable Slow Strategy

To skip the triangle breakout strategy (takes 34+ hours):

```yaml
strategies:
  triangle_breakout:
    enabled: false
```

### Disable Caching

To always run fresh (no cache):

```yaml
cache:
  enabled: false
```

### Change Cache Expiry

To keep cache forever:

```yaml
cache:
  cache_expiry_days: 0  # Never expire
```

To expire after 1 day:

```yaml
cache:
  cache_expiry_days: 1
```

## Cache Management

### Clear Cache Manually

Delete cache directory:
```bash
rm -rf cache/strategy_results/
```

### View Cache Files

List cached strategies:
```bash
ls -lh cache/strategy_results/
```

### Cache File Format

Each cache file contains:
- Strategy name and description
- Performance metrics
- All trades (as JSON records)
- Cache timestamp

## Best Practices

1. **Keep triangle_breakout disabled** unless you have 34+ hours to wait
2. **Enable caching** for faster reruns
3. **Set appropriate expiry** based on how often you change data/config
4. **Clear cache** when you update strategy code (not just config)

## Troubleshooting

### Cache Not Working

- Check `cache.enabled` is `true` in config
- Verify cache directory exists and is writable
- Check cache hasn't expired

### Wrong Results from Cache

- Clear cache: `rm -rf cache/strategy_results/`
- Or disable cache temporarily

### Strategy Still Running Despite Cache

- Verify cache key matches (same data + config)
- Check cache file exists in `cache/strategy_results/`
- Verify cache hasn't expired

