# CSCE 642: Deep RL Stock Trading with PPO

**Team Members:**
- Tanishq Chopra (436000948)
- Simran Kaur (335006983)

## Project Overview

Implementation of a Proximal Policy Optimization (PPO) agent from scratch for single-stock trading using daily OHLCV data. The project includes a custom Gym-style trading environment and comprehensive baseline comparisons.

## Project Structure

```
├── data/                   # Cached stock data
├── src/
│   ├── environment/       # Trading environment
│   ├── agents/            # PPO implementation
│   ├── baselines/         # Baseline strategies
│   ├── utils/             # Data processing, metrics
│   └── config/            # Configuration files
├── experiments/           # Training runs and results
├── notebooks/             # Analysis and visualization
└── tests/                 # Unit tests
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

## Metrics

- Cumulative Return (CAGR)
- Sharpe Ratio
- Maximum Drawdown
- Turnover
- % Profitable Days

## Baselines

- Buy-and-Hold
- Random Policy
- Moving Average Crossover

## Stretch Goals

- Transaction costs and slippage
- Short selling with risk controls
- Continuous action space
- Soft Actor-Critic (SAC) comparison
- Multi-asset portfolio (5-10 tickers)

pip install -r requirements.txt
python3 test_setup.py
python3 quick_start.py
python3 train.py --config configs/ppo_spy.yaml
python3 evaluate.py --config configs/ppo_spy.yaml --checkpoint experiments/*/final_model.pth --split test