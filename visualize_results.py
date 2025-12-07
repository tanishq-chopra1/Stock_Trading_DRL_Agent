
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')

from baselines import evaluate_all_baselines
from utils import prepare_data_for_trading
from utils.visualization import plot_training_curves, create_summary_report

def main():
    print("="*60)
    print("Generating Visualizations")
    print("="*60)
    
    # 1. Setup Configuration
    config_path = 'configs/ppo_spy.yaml'
    # Updated to point to the new experiment directory
    exp_dir = Path('experiments/ppo_spy_base_20251206_120220')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"Log directory: {exp_dir}")
    
    # 2. Plot Training Curves (Portfolio Return vs Episodes)
    log_file = exp_dir / 'training_log.csv'
    if log_file.exists():
        print(f"\nStep 1: Plotting training curves from {log_file}...")
        plot_training_curves(
            str(log_file),
            metrics=['total_return', 'sharpe_ratio'],
            title="PPO Agent Training Progress",
            save_path=exp_dir / "training_curves.png"
        )
    else:
        print("\nStep 1: Skipped training curves (log file not found)")

    # 3. Model Comparison (Run on Test Set)
    print("\nStep 2: Running baseline comparison on Test Set...")
    
    # Load Data
    _, _, test_df = prepare_data_for_trading(
        ticker=config['ticker'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        data_dir=config['data_dir'],
        add_indicators=config['add_indicators'],
        train_pct=config.get('train_pct', 0.7),
        val_pct=config.get('val_pct', 0.15),
        test_pct=config.get('test_pct', 0.15)
    )
    
    # Run Baselines
    print("Running baseline strategies...")
    results = evaluate_all_baselines(
        test_df, 
        initial_balance=config['initial_balance'],
        shares_per_trade=config['shares_per_trade']
    )
    
    # Run PPO Agent (Final Model)
    from agents import PPOAgent
    from environment import TradingEnv
    from train import evaluate_agent
    import torch
    
    print("Evaluating trained PPO agent...")
    
    # Setup Env
    env_kwargs = {
        'window_size': config['window_size'],
        'initial_balance': config['initial_balance'],
        'shares_per_trade': config['shares_per_trade'],
        'transaction_cost_pct': config['transaction_cost_pct'],
        'max_shares': config['max_shares'],
        'features': config['features'],
        'normalize': config['normalize']
    }
    env = TradingEnv(test_df, **env_kwargs)
    
    # Load Agent
    device = 'cuda' if torch.cuda.is_available() and config.get('use_cuda', False) else 'cpu'
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=config['lr'],
        hidden_dims=config['hidden_dims'],
        dropout=config.get('dropout', 0.0),
        device=device
    )
    
    model_path = exp_dir / 'final_model.pth'
    if model_path.exists():
        agent.load(str(model_path))
        
        state, _ = env.reset()
        done = False
        while not done:
            action, _, _ = agent.select_action(state, deterministic=True)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
        results['PPO Agent'] = np.array(env.portfolio_values)
        print("Agent evaluation complete.")
        
    else:
        print(f"Warning: Model not found at {model_path}")

    # 4. Generate Comprehensive Report
    print("\nStep 3: Creating visual report...")
    
    # Prepare comparison dataframe for metrics chart
    metrics_data = []
    for name, values in results.items():
        total_ret = (values[-1] / values[0]) - 1
        daily_ret = np.diff(values) / values[:-1]
        if len(daily_ret) > 1 and np.std(daily_ret) > 0:
            sharpe = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252)
        else:
            sharpe = 0.0
            
        cum_max = np.maximum.accumulate(values)
        dd = (values - cum_max) / cum_max
        max_dd = np.min(dd)
        
        metrics_data.append({
            'Strategy': name,
            'total_return': total_ret,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd
        })
    
    comparison_df = pd.DataFrame(metrics_data).set_index('Strategy')
    
    # Create plots
    create_summary_report(
        results, 
        comparison_df, 
        output_dir=str(exp_dir / 'visualizations')
    )
    
    print("\n" + "="*60)
    print("Done! Visualizations saved to:")
    print(f"  {exp_dir}/training_curves.png")
    print(f"  {exp_dir}/visualizations/")
    print("="*60)

if __name__ == "__main__":
    main()
