"""
Quick script to verify the project setup.
"""

import os
import sys
from pathlib import Path

def verify_setup():
    """Verify that all required files and directories exist."""
    project_root = Path(__file__).parent
    
    print("Verifying project setup...\n")
    
    # Check directories
    required_dirs = ['data', 'src', 'results']
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/ directory exists")
        else:
            print(f"✗ {dir_name}/ directory missing")
    
    # Check data file
    data_file = project_root / 'data' / 'nifty_1min.csv'
    if data_file.exists():
        size_mb = data_file.stat().st_size / (1024 * 1024)
        print(f"✓ Data file exists ({size_mb:.1f} MB)")
    else:
        print(f"✗ Data file missing: {data_file}")
        print("  Please copy your NIFTY 50_minute.csv to data/nifty_1min.csv")
    
    # Check source files
    required_files = [
        'src/indicators.py',
        'src/strategy.py',
        'src/backtester.py',
        'src/utils.py',
        'src/run_backtest.py',
        'config.yaml',
        'requirements.txt',
        'README.md'
    ]
    
    print("\nChecking source files...")
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"✓ {file_name}")
        else:
            print(f"✗ {file_name} missing")
    
    # Check Python dependencies
    print("\nChecking Python dependencies...")
    try:
        import pandas
        print(f"✓ pandas {pandas.__version__}")
    except ImportError:
        print("✗ pandas not installed")
    
    try:
        import numpy
        print(f"✓ numpy {numpy.__version__}")
    except ImportError:
        print("✗ numpy not installed")
    
    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
    except ImportError:
        print("✗ matplotlib not installed")
    
    try:
        import yaml
        print(f"✓ pyyaml installed")
    except ImportError:
        print("✗ pyyaml not installed")
    
    print("\n" + "="*50)
    print("Setup verification complete!")
    print("="*50)
    print("\nTo run the backtest:")
    print("  python src/run_backtest.py")

if __name__ == '__main__':
    verify_setup()

