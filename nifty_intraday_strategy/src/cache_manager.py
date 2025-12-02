"""
Cache manager for storing and retrieving calculated indicators and patterns.
Uses parameter hashing to create unique cache keys.
"""

import pandas as pd
import numpy as np
import hashlib
import json
import pickle
import os
from pathlib import Path
from typing import Optional, Dict, Any


class CacheManager:
    """
    Manages caching of calculated indicators and patterns.
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _hash_data_params(self, data_hash: str, params: Dict[str, Any]) -> str:
        """
        Create a hash from data hash and parameters.
        
        Args:
            data_hash: Hash of the data
            params: Parameters dictionary
        
        Returns:
            Combined hash string
        """
        # Sort params for consistent hashing
        params_str = json.dumps(params, sort_keys=True)
        combined = f"{data_hash}_{params_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _hash_dataframe(self, df: pd.DataFrame, sample_size: int = 1000) -> str:
        """
        Create a hash of the dataframe (using first/last rows and shape).
        
        Args:
            df: DataFrame to hash
            sample_size: Number of rows to sample for hashing
        
        Returns:
            Hash string
        """
        # Use shape, first few rows, last few rows, and column names
        shape_str = f"{df.shape[0]}_{df.shape[1]}"
        
        # Sample first and last rows
        if len(df) > sample_size:
            sample = pd.concat([
                df.head(sample_size // 2),
                df.tail(sample_size // 2)
            ])
        else:
            sample = df
        
        # Create hash from sample
        sample_str = sample.to_string()
        cols_str = "_".join(df.columns.tolist())
        
        combined = f"{shape_str}_{cols_str}_{sample_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_cache_path(self, cache_key: str, indicator_name: str) -> Path:
        """
        Get cache file path for an indicator.
        
        Args:
            cache_key: Unique cache key
            indicator_name: Name of the indicator
        
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{indicator_name}_{cache_key}.pkl"
    
    def load_from_cache(self, cache_key: str, indicator_name: str) -> Optional[pd.DataFrame]:
        """
        Load indicator/pattern from cache.
        
        Args:
            cache_key: Unique cache key
            indicator_name: Name of the indicator
        
        Returns:
            Cached DataFrame or None if not found
        """
        cache_path = self.get_cache_path(cache_key, indicator_name)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"  Warning: Could not load cache for {indicator_name}: {e}")
                return None
        
        return None
    
    def save_to_cache(self, data: pd.DataFrame, cache_key: str, indicator_name: str):
        """
        Save indicator/pattern to cache.
        
        Args:
            data: DataFrame to cache
            cache_key: Unique cache key
            indicator_name: Name of the indicator
        """
        cache_path = self.get_cache_path(cache_key, indicator_name)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"  Warning: Could not save cache for {indicator_name}: {e}")
    
    def get_cache_key(self, df: pd.DataFrame, params: Dict[str, Any], 
                     indicator_name: str) -> str:
        """
        Get cache key for given data and parameters.
        
        Args:
            df: DataFrame
            params: Parameters dictionary
            indicator_name: Name of the indicator
        
        Returns:
            Cache key string
        """
        data_hash = self._hash_dataframe(df)
        return self._hash_data_params(data_hash, params)
    
    def clear_cache(self, indicator_name: Optional[str] = None):
        """
        Clear cache files.
        
        Args:
            indicator_name: If provided, clear only this indicator's cache.
                           If None, clear all cache.
        """
        if indicator_name:
            pattern = f"{indicator_name}_*.pkl"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
        else:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
        print(f"Cache cleared for {indicator_name or 'all indicators'}")


def cached_indicator(func):
    """
    Decorator to cache indicator calculation results.
    
    Usage:
        @cached_indicator
        def calculate_something(df, param1, param2):
            ...
    """
    def wrapper(df: pd.DataFrame, cache_manager: Optional[CacheManager] = None, 
                *args, **kwargs):
        # Extract cache_manager if provided
        if cache_manager is None:
            # Try to get from kwargs
            cache_manager = kwargs.pop('cache_manager', None)
        
        # If no cache manager, just run the function
        if cache_manager is None:
            return func(df, *args, **kwargs)
        
        # Create cache key
        params = {**kwargs}
        if args:
            # Add positional args as well if needed
            params['_args'] = args
        
        indicator_name = func.__name__
        cache_key = cache_manager.get_cache_key(df, params, indicator_name)
        
        # Try to load from cache
        cached_result = cache_manager.load_from_cache(cache_key, indicator_name)
        if cached_result is not None:
            print(f"  ✓ Loaded {indicator_name} from cache")
            return cached_result
        
        # Calculate
        print(f"  Calculating {indicator_name}...", end='', flush=True)
        result = func(df, *args, **kwargs)
        print(" ✓")
        
        # Save to cache
        cache_manager.save_to_cache(result, cache_key, indicator_name)
        
        return result
    
    return wrapper

