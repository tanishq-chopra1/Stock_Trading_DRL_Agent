"""
Baseline trading strategies for comparison
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


class BaselineStrategy:
    """Base class for baseline strategies"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        """Reset strategy state"""
        self.cash = self.initial_balance
        self.shares = 0
        self.portfolio_values = []
        self.trades = []
    
    def get_action(self, data: pd.DataFrame, current_idx: int) -> int:
        """
        Get action for current step
        
        Args:
            data: Full dataframe
            current_idx: Current index in dataframe
            
        Returns:
            action: 0=Buy, 1=Hold, 2=Sell
        """
        raise NotImplementedError
    
    def execute_trade(self, action: int, price: float, shares_per_trade: int = 10):
        """Execute trade based on action"""
        if action == 0:  # Buy
            max_buyable = int(self.cash / price)
            shares_to_buy = min(shares_per_trade, max_buyable)
            if shares_to_buy > 0:
                self.cash -= shares_to_buy * price
                self.shares += shares_to_buy
                
        elif action == 2:  # Sell
            shares_to_sell = min(shares_per_trade, self.shares)
            if shares_to_sell > 0:
                self.cash += shares_to_sell * price
                self.shares -= shares_to_sell
    
    def get_portfolio_value(self, price: float) -> float:
        """Calculate current portfolio value"""
        return self.cash + self.shares * price
    
    def run(
        self, 
        data: pd.DataFrame, 
        shares_per_trade: int = 10
    ) -> np.ndarray:
        """
        Run strategy on data
        
        Args:
            data: DataFrame with price data
            shares_per_trade: Number of shares per trade
            
        Returns:
            Array of portfolio values
        """
        self.reset()
        
        for idx in range(len(data)):
            price = data.iloc[idx]['close']
            
            # Get action from strategy
            action = self.get_action(data, idx)
            
            # Execute trade
            self.execute_trade(action, price, shares_per_trade)
            
            # Record portfolio value
            portfolio_value = self.get_portfolio_value(price)
            self.portfolio_values.append(portfolio_value)
        
        return np.array(self.portfolio_values)


class BuyAndHold(BaselineStrategy):
    """Buy and hold strategy - buy at start, hold until end"""
    
    def __init__(self, initial_balance: float = 10000.0):
        super().__init__(initial_balance)
        self.bought = False
    
    def reset(self):
        super().reset()
        self.bought = False
    
    def get_action(self, data: pd.DataFrame, current_idx: int) -> int:
        """Buy on first step, hold afterwards"""
        if not self.bought and current_idx == 0:
            self.bought = True
            return 0  # Buy
        return 1  # Hold


class RandomPolicy(BaselineStrategy):
    """Random action selection"""
    
    def __init__(self, initial_balance: float = 10000.0, seed: int = None):
        super().__init__(initial_balance)
        self.rng = np.random.RandomState(seed)
    
    def get_action(self, data: pd.DataFrame, current_idx: int) -> int:
        """Select random action"""
        return self.rng.choice([0, 1, 2])


class MovingAverageCrossover(BaselineStrategy):
    """
    Simple Moving Average Crossover Strategy
    Buy when short MA crosses above long MA
    Sell when short MA crosses below long MA
    """
    
    def __init__(
        self, 
        initial_balance: float = 10000.0,
        short_window: int = 10,
        long_window: int = 30
    ):
        super().__init__(initial_balance)
        self.short_window = short_window
        self.long_window = long_window
    
    def get_action(self, data: pd.DataFrame, current_idx: int) -> int:
        """Generate signal based on MA crossover"""
        # Need enough history for long MA
        if current_idx < self.long_window:
            return 1  # Hold
        
        # Calculate moving averages
        prices = data.iloc[:current_idx + 1]['close'].values
        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])
        
        # Previous MAs
        if current_idx > self.long_window:
            prev_prices = data.iloc[:current_idx]['close'].values
            prev_short_ma = np.mean(prev_prices[-self.short_window:])
            prev_long_ma = np.mean(prev_prices[-self.long_window:])
            
            # Crossover detection
            if prev_short_ma <= prev_long_ma and short_ma > long_ma:
                # Bullish crossover - buy
                return 0
            elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
                # Bearish crossover - sell
                return 2
        
        # No crossover - maintain position
        if self.shares > 0:
            return 1  # Hold if we have shares
        else:
            # If short MA above long MA and we have no shares, buy
            if short_ma > long_ma:
                return 0
            else:
                return 1


class MomentumStrategy(BaselineStrategy):
    """
    Momentum strategy based on recent returns
    Buy if recent return is positive, sell if negative
    """
    
    def __init__(
        self, 
        initial_balance: float = 10000.0,
        lookback: int = 5,
        threshold: float = 0.0
    ):
        super().__init__(initial_balance)
        self.lookback = lookback
        self.threshold = threshold
    
    def get_action(self, data: pd.DataFrame, current_idx: int) -> int:
        """Generate signal based on momentum"""
        if current_idx < self.lookback:
            return 1  # Hold
        
        # Calculate recent return
        prices = data.iloc[:current_idx + 1]['close'].values
        recent_return = (prices[-1] / prices[-self.lookback]) - 1
        
        if recent_return > self.threshold:
            return 0  # Buy on positive momentum
        elif recent_return < -self.threshold:
            return 2  # Sell on negative momentum
        else:
            return 1  # Hold


class RSIStrategy(BaselineStrategy):
    """
    RSI-based strategy
    Buy when RSI < oversold threshold
    Sell when RSI > overbought threshold
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        rsi_period: int = 14,
        oversold: float = 30,
        overbought: float = 70
    ):
        super().__init__(initial_balance)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, prices: np.ndarray) -> float:
        """Calculate RSI"""
        if len(prices) < self.rsi_period + 1:
            return 50.0  # Neutral
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_action(self, data: pd.DataFrame, current_idx: int) -> int:
        """Generate signal based on RSI"""
        if current_idx < self.rsi_period:
            return 1  # Hold
        
        prices = data.iloc[:current_idx + 1]['close'].values
        rsi = self.calculate_rsi(prices)
        
        if rsi < self.oversold:
            return 0  # Buy when oversold
        elif rsi > self.overbought:
            return 2  # Sell when overbought
        else:
            return 1  # Hold


def evaluate_all_baselines(
    data: pd.DataFrame,
    initial_balance: float = 10000.0,
    shares_per_trade: int = 10
) -> dict:
    """
    Evaluate all baseline strategies on given data
    
    Args:
        data: DataFrame with price data
        initial_balance: Starting cash
        shares_per_trade: Shares per trade
        
    Returns:
        Dictionary mapping strategy name to portfolio values
    """
    strategies = {
        'Buy & Hold': BuyAndHold(initial_balance),
        'Random': RandomPolicy(initial_balance, seed=42),
        'MA Crossover (10/30)': MovingAverageCrossover(initial_balance, 10, 30),
        'MA Crossover (20/50)': MovingAverageCrossover(initial_balance, 20, 50),
        'Momentum (5d)': MomentumStrategy(initial_balance, lookback=5),
        'RSI (14)': RSIStrategy(initial_balance, rsi_period=14)
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"Running {name}...")
        portfolio_values = strategy.run(data, shares_per_trade)
        results[name] = portfolio_values
    
    return results
