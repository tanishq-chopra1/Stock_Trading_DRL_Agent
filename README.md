# CSCE 642: Deep RL Stock Trading with PPO

**Team Members:**
- Tanishq Chopra (436000948)
- Simran Kaur (335006983)

## Project Overview

Implementation of a Proximal Policy Optimization (PPO) agent from scratch for single-stock trading using daily OHLCV data. The project includes a custom Gym-style trading environment and comprehensive baseline comparisons.

## Project Structure

```
├── configs/           # Configuration files
├── data/              # Stock data
├── experiments/       # Training logs and checkpoints
├── configs/           # Configuration files
├── data/              # Stock data
├── experiments/       # Training logs and checkpoints
├── src/
│   ├── agents/        # PPO Agent implementation
│   ├── baselines/     # Trading baselines
│   ├── environment/   # Custom Trading Environment
│   └── utils/         # Helper functions
├── evaluate.py        # Evaluation script
├── download_data.py   # Data download script
├── quick_start.py     # Automates setup and training
├── test_setup.py      # Environment verification script
├── train.py           # Training script
└── visualize_results.py # Visualization script
│   ├── agents/        # PPO Agent implementation
│   ├── baselines/     # Trading baselines
│   ├── environment/   # Custom Trading Environment
│   └── utils/         # Helper functions
├── evaluate.py        # Evaluation script
├── download_data.py   # Data download script
├── quick_start.py     # Automates setup and training
├── test_setup.py      # Environment verification script
├── train.py           # Training script
└── visualize_results.py # Visualization script
```

## Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. Download data:
```bash
python3 src/utils/data_utils.py --ticker SPY --start 2010-01-01 --end 2020-01-01
```

2. Train PPO agent:
```bash
python3 train.py --config configs/ppo_spy.yaml
```

3. Evaluate:
```bash
python3 evaluate.py --checkpoint experiments/ppo_spy_base_20251206_120220/final_model.pth --split test --deterministic
```

4. Visualize Results:
```bash
python3 visualize_results.py
```

4. Visualize Results:
```bash
python3 visualize_results.py
```

## Metrics

- Cumulative Return (CAGR)
- Sharpe Ratio
- Maximum Drawdown
- Turnover
- % Profitable Days

## Baselines

- Buy-and-Hold
- Random Policy
- Moving Average Crossover (SMA)
- Momentum Strategy
- RSI Strategy

## Results

 The PPO agent achieved a cumulative return of **20.03%** on the unseen test set (July 2018 – Dec 2019), significantly outperforming the Buy-and-Hold baseline (5.31%) and minimizing drawdowns compared to the market.

### Artifacts Location
*   **Visualizations** (Plots for Portfolio Value, Drawdowns, etc.) are generated in:
    `experiments/<experiment_name>/visualizations/`
*   **Training Logs** (CSV of per-episode metrics) are saved to:
    `experiments/<experiment_name>/training_log.csv`
*   **Trained Models** (PyTorch checkpoints) are stored as:
    `experiments/<experiment_name>/final_model.pth`

## Stretch Goals

- Transaction costs and slippage
- Short selling with risk controls
- Continuous action space
- Soft Actor-Critic (SAC) comparison
- Multi-asset portfolio (5-10 tickers)
