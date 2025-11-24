"""
Quick start script to download data and run a simple test
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Quick start workflow"""
    print("\n" + "="*60)
    print("CSCE 642: Deep RL Stock Trading - Quick Start")
    print("="*60 + "\n")
    
    # Step 1: Test setup
    print("Step 1: Testing project setup...")
    print("-" * 60)
    result = subprocess.run([sys.executable, 'test_setup.py'], capture_output=False)
    if result.returncode != 0:
        print("✗ Setup tests failed. Please check the errors above.")
        return
    
    # Step 2: Download data
    print("\n\nStep 2: Downloading SPY data...")
    print("-" * 60)
    result = subprocess.run([
        sys.executable, 
        'src/utils/data_utils.py',
        '--ticker', 'SPY',
        '--start', '2014-01-01',
        '--end', '2024-01-01',
        '--add_indicators'
    ], capture_output=False)
    
    if result.returncode != 0:
        print("✗ Data download failed. Check your internet connection.")
        return
    
    # Step 3: Instructions for training
    print("\n\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nYour project is ready. Here's what you can do next:\n")
    
    print("🎯 Train the PPO agent:")
    print("   python train.py --config configs/ppo_spy.yaml\n")
    
    print("📊 Evaluate the trained model:")
    print("   python evaluate.py --config configs/ppo_spy.yaml --checkpoint experiments/*/best_model.pth\n")
    
    print("🔬 Try different configurations:")
    print("   - configs/ppo_spy.yaml (basic)")
    print("   - configs/ppo_spy_indicators.yaml (with technical indicators)")
    print("   - configs/ppo_aapl.yaml (different stock)\n")
    
    print("📝 Available data:")
    data_dir = Path('data')
    if data_dir.exists():
        csv_files = list(data_dir.glob('*.csv'))
        for f in csv_files:
            print(f"   - {f.name}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
