# Example: Complete Trading Workflow

This notebook demonstrates the complete workflow for training and evaluating a PPO agent for stock trading.

## Setup

```python
import sys
sys.path.append('../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment import TradingEnv
from agents import PPOAgent
from baselines import evaluate_all_baselines
from utils import prepare_data_for_trading, calculate_all_metrics, compare_strategies, print_metrics
```

## 1. Load and Prepare Data

```python
# Download and prepare data
train_df, val_df, test_df = prepare_data_for_trading(
    ticker='SPY',
    start_date='2014-01-01',
    end_date='2024-01-01',
    data_dir='../data',
    add_indicators=False,
    train_pct=0.7,
    val_pct=0.15,
    test_pct=0.15
)

print(f"Train: {len(train_df)} days")
print(f"Val:   {len(val_df)} days")
print(f"Test:  {len(test_df)} days")
```

## 2. Create Environment

```python
env_kwargs = {
    'window_size': 30,
    'initial_balance': 10000.0,
    'shares_per_trade': 10,
    'transaction_cost_pct': 0.0,
    'max_shares': 1000,
    'features': ['close', 'volume'],
    'normalize': True
}

train_env = TradingEnv(train_df, **env_kwargs)
val_env = TradingEnv(val_df, **env_kwargs)
test_env = TradingEnv(test_df, **env_kwargs)

print(f"Observation space: {train_env.observation_space.shape}")
print(f"Action space: {train_env.action_space.n}")
```

## 3. Create PPO Agent

```python
agent = PPOAgent(
    state_dim=train_env.observation_space.shape[0],
    action_dim=train_env.action_space.n,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    hidden_dims=[256, 128],
    device='cuda'  # or 'cpu'
)

print("Agent created successfully!")
```

## 4. Training Loop (Simple Example)

```python
n_episodes = 50  # Use 500+ for real training
val_frequency = 10

for episode in range(n_episodes):
    # Training episode
    state, _ = train_env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = train_env.step(action)
        done = terminated or truncated
        agent.store_reward_and_done(reward, done)
        state = next_state
    
    # Update policy
    metrics = agent.update(state, n_epochs=10, batch_size=64)
    train_stats = train_env.get_episode_stats()
    
    # Validation
    if (episode + 1) % val_frequency == 0:
        state, _ = val_env.reset()
        done = False
        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, _, terminated, truncated, _ = val_env.step(action)
            done = terminated or truncated
            state = next_state
        
        val_stats = val_env.get_episode_stats()
        
        print(f"Episode {episode+1}/{n_episodes}")
        print(f"  Train - Return: {train_stats['total_return']:.2%}, Sharpe: {train_stats['sharpe_ratio']:.2f}")
        print(f"  Val   - Return: {val_stats['total_return']:.2%}, Sharpe: {val_stats['sharpe_ratio']:.2f}")
```

## 5. Evaluate on Test Set

```python
# Run PPO agent
state, _ = test_env.reset()
done = False

while not done:
    action = agent.select_action(state, deterministic=True)
    next_state, _, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated
    state = next_state

ppo_stats = test_env.get_episode_stats()
ppo_values = np.array(test_env.portfolio_values)

print_metrics(ppo_stats, "PPO Performance")
```

## 6. Compare with Baselines

```python
# Run all baseline strategies
baseline_results = evaluate_all_baselines(
    test_df,
    initial_balance=10000.0,
    shares_per_trade=10
)

# Add PPO results
all_results = {'PPO': ppo_values, **baseline_results}

# Create comparison table
comparison_df = compare_strategies(all_results)
print(comparison_df)
```

## 7. Visualize Results

```python
# Plot portfolio values
plt.figure(figsize=(14, 8))
for name, values in all_results.items():
    normalized = (values / values[0]) * 100
    plt.plot(normalized, label=name, linewidth=2)

plt.xlabel('Trading Days')
plt.ylabel('Portfolio Value (Normalized to 100)')
plt.title('Strategy Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 8. Analyze Returns

```python
# Calculate daily returns for each strategy
returns_dict = {}
for name, values in all_results.items():
    returns = np.diff(values) / values[:-1]
    returns_dict[name] = returns

# Plot return distributions
fig, axes = plt.subplots(1, len(returns_dict), figsize=(20, 4))
for (name, returns), ax in zip(returns_dict.items(), axes):
    ax.hist(returns, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--')
    ax.set_title(name)
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()
```

## 9. Risk Analysis

```python
# Maximum drawdown analysis
for name, values in all_results.items():
    cummax = np.maximum.accumulate(values)
    drawdown = (values - cummax) / cummax
    max_dd = np.min(drawdown)
    print(f"{name:20s}: Max Drawdown = {max_dd:.2%}")
```

## 10. Save Model

```python
# Save trained model
agent.save('../experiments/notebook_model.pth')
print("Model saved!")

# To load later:
# agent.load('../experiments/notebook_model.pth')
```

---

## Key Takeaways

1. **Data Preparation**: Use chronological splits for time series
2. **Environment**: Gym-compatible interface makes testing easy
3. **Training**: PPO is stable but requires proper hyperparameters
4. **Evaluation**: Always compare against strong baselines
5. **Metrics**: Use risk-adjusted metrics (Sharpe) not just returns

## Next Steps

- Run with more episodes (500-1000)
- Try different stocks (AAPL, MSFT, etc.)
- Add technical indicators
- Experiment with transaction costs
- Implement stretch goals (SAC, multi-asset, etc.)
