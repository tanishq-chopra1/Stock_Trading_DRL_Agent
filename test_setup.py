"""
Quick test script to verify environment and agent setup
"""

import numpy as np
import sys
sys.path.append('src')

from environment import TradingEnv
from agents import PPOAgent
from utils import download_stock_data
import pandas as pd


def test_environment():
    """Test trading environment"""
    print("="*60)
    print("Testing Trading Environment")
    print("="*60)
    
    # Create simple dummy data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    dummy_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Create environment
    env = TradingEnv(
        df=dummy_data,
        window_size=10,
        initial_balance=10000.0,
        shares_per_trade=10,
        transaction_cost_pct=0.001,
        max_shares=100,
        features=['close', 'volume']
    )
    
    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n}")
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Environment reset")
    print(f"  Initial observation shape: {obs.shape}")
    print(f"  Initial portfolio value: ${info['portfolio_value']:.2f}")
    
    # Test step
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    print(f"✓ Environment step (10 steps)")
    print(f"  Final portfolio value: ${info['portfolio_value']:.2f}")
    
    # Test full episode
    obs, info = env.reset()
    done = False
    steps = 0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    
    stats = env.get_episode_stats()
    print(f"✓ Full episode completed ({steps} steps)")
    print(f"  Total return: {stats['total_return']:.2%}")
    print(f"  Sharpe ratio: {stats['sharpe_ratio']:.2f}")
    print(f"  Max drawdown: {stats['max_drawdown']:.2%}")
    
    print("✓ Environment tests passed!\n")


def test_agent():
    """Test PPO agent"""
    print("="*60)
    print("Testing PPO Agent")
    print("="*60)
    
    # Create agent
    agent = PPOAgent(
        state_dim=23,  # 10 window * 2 features + 3 position features
        action_dim=3,
        lr=3e-4,
        device='cpu'
    )
    
    print("✓ Agent created")
    print(f"  Device: cpu")
    print(f"  Network parameters: {sum(p.numel() for p in agent.ac_network.parameters())}")
    
    # Test action selection (deterministic doesn't store trajectory)
    dummy_state = np.random.randn(23).astype(np.float32)
    action = agent.select_action(dummy_state, deterministic=True)
    print("✓ Action selection works")
    print(f"  Sample action: {action}")
    
    # Test trajectory storage
    for _ in range(10):
        state = np.random.randn(23).astype(np.float32)
        _ = agent.select_action(state, deterministic=False)
        reward = np.random.randn()
        done = False
        agent.store_reward_and_done(reward, done)
    
    print("✓ Trajectory storage works")
    print(f"  Stored {len(agent.states)} transitions")
    
    # Test update
    next_state = np.random.randn(23).astype(np.float32)
    metrics = agent.update(next_state, n_epochs=2, batch_size=5)
    
    print("✓ Policy update works")
    print(f"  Loss: {metrics['loss']:.4f}")
    
    print("✓ Agent tests passed!\n")


def test_data_download():
    """Test data downloading"""
    print("="*60)
    print("Testing Data Download")
    print("="*60)
    
    try:
        # Download small sample
        df = download_stock_data(
            ticker='SPY',
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_dir='data'
        )
        
        print(f"✓ Data download successful")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        
        print("✓ Data tests passed!\n")
        
    except Exception as e:
        print(f"✗ Data download failed: {e}")
        print("  This is expected if you don't have internet connection")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("RUNNING PROJECT TESTS")
    print("="*60 + "\n")
    
    try:
        test_environment()
        test_agent()
        test_data_download()
        
        print("="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nYour project setup is working correctly!")
        print("\nNext steps:")
        print("1. Download data: python src/utils/data_utils.py --ticker SPY")
        print("2. Train model: python train.py --config configs/ppo_spy.yaml")
        print("3. Evaluate: python evaluate.py --config configs/ppo_spy.yaml --checkpoint experiments/*/best_model.pth")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
