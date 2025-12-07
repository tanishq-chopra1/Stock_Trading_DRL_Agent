"""
Training script for PPO agent on stock trading
"""

import torch
import numpy as np
import pandas as pd
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import sys
sys.path.append('src')

from environment import TradingEnv
from agents import PPOAgent
from utils import prepare_data_for_trading, calculate_all_metrics, print_metrics


def train_ppo(config: dict):
    """
    Train PPO agent with given configuration
    
    Args:
        config: Configuration dictionary
    """
    # Set random seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path('experiments') / f"{config['exp_name']}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f"Experiment directory: {exp_dir}")
    print(f"Configuration: {config}")
    
    # Prepare data
    print("\n" + "="*60)
    print("Loading and preparing data...")
    print("="*60)
    
    train_df, val_df, test_df = prepare_data_for_trading(
        ticker=config['ticker'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        data_dir=config['data_dir'],
        add_indicators=config['add_indicators'],
        train_pct=config['train_pct'],
        val_pct=config['val_pct'],
        test_pct=config['test_pct']
    )
    
    # Create environments
    env_kwargs = {
        'window_size': config['window_size'],
        'initial_balance': config['initial_balance'],
        'shares_per_trade': config['shares_per_trade'],
        'transaction_cost_pct': config['transaction_cost_pct'],
        'max_shares': config['max_shares'],
        'features': config['features'],
        'normalize': config['normalize']
    }
    
    train_env = TradingEnv(train_df, **env_kwargs)
    val_env = TradingEnv(val_df, **env_kwargs)
    
    print(f"\nEnvironment observation space: {train_env.observation_space.shape}")
    print(f"Environment action space: {train_env.action_space.n}")
    
    # Create agent
    device = 'cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu'
    print(f"\nUsing device: {device}")
    
    agent = PPOAgent(
        state_dim=train_env.observation_space.shape[0],
        action_dim=train_env.action_space.n,
        lr=config['lr'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_epsilon=config['clip_epsilon'],
        value_coef=config['value_coef'],
        entropy_coef=config['entropy_coef'],
        max_grad_norm=config['max_grad_norm'],
        hidden_dims=config['hidden_dims'],
        dropout=config.get('dropout', 0.0),  # Add dropout support
        device=device
    )
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_val_sharpe = -np.inf
    patience_counter = 0
    training_logs = []
    
    for episode in range(config['n_episodes']):
        # Training episode
        state, _ = train_env.reset()
        
        # Collect trajectory
        done = False
        while not done:
            action, _, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
            
            agent.store_reward(reward, done)
            state = next_state
        
        # Update policy
        metrics = agent.update(
            n_epochs=config['ppo_epochs'],
            batch_size=config['batch_size']
        )
        
        # Get episode statistics
        train_stats = train_env.get_episode_stats()
        
        # Validation
        if (episode + 1) % config['val_frequency'] == 0:
            val_stats = evaluate_agent(agent, val_env, deterministic=True)
            
            # Print progress
            print(f"\nEpisode {episode + 1}/{config['n_episodes']}")
            print(f"Train - Return: {train_stats['total_return']:.2%}, Sharpe: {train_stats['sharpe_ratio']:.2f}")
            print(f"Val   - Return: {val_stats['total_return']:.2%}, Sharpe: {val_stats['sharpe_ratio']:.2f}")
            print(f"Loss: {metrics['total_loss']:.4f}, Policy: {metrics['policy_loss']:.4f}, Value: {metrics['value_loss']:.4f}")
            
            # Log metrics
            log_entry = {
                'episode': episode + 1,
                'train_total_return': train_stats['total_return'],
                'train_sharpe_ratio': train_stats['sharpe_ratio'],
                'val_total_return': val_stats['total_return'],
                'val_sharpe_ratio': val_stats['sharpe_ratio'],
                'policy_loss': metrics['policy_loss'],
                'value_loss': metrics['value_loss'],
                'total_loss': metrics['total_loss']
            }
            training_logs.append(log_entry)
            
            # Save logs to CSV
            pd.DataFrame(training_logs).to_csv(exp_dir / 'training_log.csv', index=False)
            
            # Early stopping based on validation Sharpe ratio
            if val_stats['sharpe_ratio'] > best_val_sharpe:
                best_val_sharpe = val_stats['sharpe_ratio']
                patience_counter = 0
                
                # Save best model
                best_model_path = exp_dir / 'best_model.pth'
                agent.save(best_model_path)
                print(f"✓ New best model saved (Sharpe: {best_val_sharpe:.2f})")
            else:
                patience_counter += 1
                
            # Check early stopping
            if patience_counter >= config['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {episode + 1} episodes")
                break
    
    # Save final model
    final_model_path = exp_dir / 'final_model.pth'
    agent.save(final_model_path)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation Sharpe ratio: {best_val_sharpe:.2f}")
    print(f"Models saved to {exp_dir}")
    print("="*60)
    
    return exp_dir


def evaluate_agent(agent: PPOAgent, env: TradingEnv, deterministic: bool = True) -> dict:
    """
    Evaluate agent on an environment
    
    Args:
        agent: PPO agent
        env: Trading environment
        deterministic: Whether to use deterministic actions
        
    Returns:
        Episode statistics
    """
    state, _ = env.reset()
    done = False
    
    while not done:
        action, _, _ = agent.select_action(state, deterministic=deterministic)
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
    
    stats = env.get_episode_stats()
    return stats


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent for stock trading')
    parser.add_argument('--config', type=str, default='configs/ppo_spy.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train
    exp_dir = train_ppo(config)
    print(f"\nExperiment completed: {exp_dir}")


if __name__ == '__main__':
    main()
