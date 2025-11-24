"""Evaluate a saved PPO model on a selected split (train/val/test).

Usage examples (PowerShell):
    python evaluate.py --config configs/ppo_spy.yaml --checkpoint experiments/ppo_spy_base_20250.../best_model.pth --split test

WSL example:
    wsl bash -c "cd '/mnt/c/Users/tanis/Desktop/Sem 3 TAMU/CSCE 642 Deep Reinforcement Learning/StockTrading DRL Project' && source venv/bin/activate && python evaluate.py --config configs/ppo_spy.yaml --checkpoint experiments/ppo_spy_base_YYYYMMDD_HHMMSS/best_model.pth --split test"

If --checkpoint is omitted, the script will try to find the latest experiments folder that matches the config exp_name and use its best_model.pth.
"""
import argparse
import yaml
import sys
from pathlib import Path

# Make sure src is importable
sys.path.append('src')

import torch
from agents import PPOAgent
from environment import TradingEnv
from utils import prepare_data_for_trading
from train import evaluate_agent


def find_latest_checkpoint(exp_name: str) -> Path:
    base = Path('experiments')
    if not base.exists():
        raise FileNotFoundError('No experiments directory found')
    # Find folders that start with exp_name
    matches = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith(exp_name)], key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f'No experiment directories found for {exp_name}')
    candidate = matches[0] / 'best_model.pth'
    if not candidate.exists():
        raise FileNotFoundError(f'No best_model.pth in {matches[0]}')
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/ppo_spy.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--split', type=str, default='test', choices=['train','val','test'])
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Determine checkpoint
    if args.checkpoint is None:
        ckpt_path = find_latest_checkpoint(config['exp_name'])
        print(f'Using checkpoint: {ckpt_path}')
    else:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    # Prepare data
    train_df, val_df, test_df = prepare_data_for_trading(
        ticker=config['ticker'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        data_dir=config['data_dir'],
        add_indicators=config['add_indicators'],
        train_pct=config.get('train_pct', 0.7),
        val_pct=config.get('val_pct', 0.15),
        test_pct=config.get('test_pct', 0.15)
    )

    env_kwargs = {
        'window_size': config['window_size'],
        'initial_balance': config['initial_balance'],
        'shares_per_trade': config['shares_per_trade'],
        'transaction_cost_pct': config['transaction_cost_pct'],
        'max_shares': config['max_shares'],
        'features': config['features'],
        'normalize': config['normalize']
    }

    if args.split == 'train':
        df = train_df
    elif args.split == 'val':
        df = val_df
    else:
        df = test_df

    env = TradingEnv(df, **env_kwargs)

    device = 'cuda' if torch.cuda.is_available() and config.get('use_cuda', False) else 'cpu'

    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=config['lr'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_epsilon=config['clip_epsilon'],
        value_coef=config['value_coef'],
        entropy_coef=config['entropy_coef'],
        max_grad_norm=config['max_grad_norm'],
        hidden_dims=config['hidden_dims'],
        dropout=config.get('dropout', 0.0),
        device=device
    )

    # Load checkpoint
    agent.load(str(ckpt_path))

    # Evaluate
    stats = evaluate_agent(agent, env, deterministic=args.deterministic)

    print('\n=== EVALUATION RESULTS ===')
    print(f"Split: {args.split}")
    print(f"Total return: {stats['total_return']*100:.2f}%")
    print(f"Final portfolio value: ${stats['final_value']:.2f}")
    print(f"Sharpe ratio: {stats['sharpe_ratio']:.2f}")
    print(f"Max drawdown: {stats['max_drawdown']:.2%}")
    print(f"Number of trades: {stats['num_trades']}")


if __name__ == '__main__':
    main()
