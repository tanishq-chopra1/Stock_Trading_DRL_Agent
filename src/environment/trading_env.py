"""
Trading Environment for Single-Stock RL Agent
Implements a Gym-style environment for trading a single equity
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class TradingEnv(gym.Env):
    """
    Single-stock trading environment with discrete actions.
    
    State: Windowed OHLCV features + position info + time index
    Actions: Buy, Hold, Sell
    Reward: Change in portfolio equity - transaction costs
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df,
        window_size: int = 30,
        initial_balance: float = 10000.0,
        shares_per_trade: int = 10,
        transaction_cost_pct: float = 0.0,
        max_shares: int = 1000,
        features: list = ['close', 'volume'],
        normalize: bool = True
    ):
        """
        Args:
            df: DataFrame with OHLCV data (index=date, columns=['open','high','low','close','volume'])
            window_size: Number of past days to include in state
            initial_balance: Starting cash
            shares_per_trade: Number of shares to buy/sell per action
            transaction_cost_pct: Transaction cost as % of trade value (0.001 = 0.1%)
            max_shares: Maximum inventory cap
            features: Which columns to use as features
            normalize: Whether to normalize features
        """
        super().__init__()
        
        self.df = df.copy()
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.shares_per_trade = shares_per_trade
        self.transaction_cost_pct = transaction_cost_pct
        self.max_shares = max_shares
        self.feature_columns = features
        self.normalize = normalize
        
        # Validate data
        assert len(self.df) > window_size, "Data too short for window size"
        assert all(col in self.df.columns for col in self.feature_columns), "Missing features"
        
        # Action space: 0=Buy, 1=Hold, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: window of features + position info
        n_features = len(self.feature_columns)
        obs_dim = window_size * n_features + 3  # +3 for shares, cash, time_index
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.current_step = 0
        self.cash = 0.0
        self.shares = 0
        self.portfolio_values = []
        self.trades = []
        
        # Compute feature statistics for normalization
        if self.normalize:
            self._compute_feature_stats()
    
    def _compute_feature_stats(self):
        """Compute mean and std for feature normalization"""
        self.feature_mean = self.df[self.feature_columns].mean().values
        self.feature_std = self.df[self.feature_columns].std().values + 1e-8
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using precomputed statistics"""
        if self.normalize:
            return (features - self.feature_mean) / self.feature_std
        return features
    
    def _get_observation(self) -> np.ndarray:
        """Construct the current state observation"""
        # Get windowed price features
        start_idx = self.current_step
        end_idx = self.current_step + self.window_size
        
        window_data = self.df.iloc[start_idx:end_idx][self.feature_columns].values
        window_features = self._normalize_features(window_data).flatten()
        
        # Current position info (normalized)
        current_price = self.df.iloc[end_idx - 1]['close']
        position_info = np.array([
            self.shares / self.max_shares,  # Normalized shares
            self.cash / self.initial_balance,  # Normalized cash
            (self.current_step) / len(self.df)  # Time progress
        ], dtype=np.float32)
        
        obs = np.concatenate([window_features, position_info])
        return obs.astype(np.float32)
    
    def _get_current_price(self) -> float:
        """Get the current close price"""
        return self.df.iloc[self.current_step + self.window_size - 1]['close']
    
    def _execute_trade(self, action: int) -> Tuple[float, int]:
        """
        Execute trade and return (transaction_cost, shares_traded)
        
        Actions:
            0: Buy
            1: Hold
            2: Sell
        """
        current_price = self._get_current_price()
        shares_traded = 0
        transaction_cost = 0.0
        
        if action == 0:  # Buy
            max_buyable = int(self.cash / current_price)
            shares_to_buy = min(self.shares_per_trade, max_buyable, 
                               self.max_shares - self.shares)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                transaction_cost = cost * self.transaction_cost_pct
                self.cash -= (cost + transaction_cost)
                self.shares += shares_to_buy
                shares_traded = shares_to_buy
                
        elif action == 2:  # Sell
            shares_to_sell = min(self.shares_per_trade, self.shares)
            
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price
                transaction_cost = revenue * self.transaction_cost_pct
                self.cash += (revenue - transaction_cost)
                self.shares -= shares_to_sell
                shares_traded = -shares_to_sell
        
        # action == 1 (Hold) does nothing
        
        return transaction_cost, shares_traded
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current total portfolio value"""
        current_price = self._get_current_price()
        return self.cash + self.shares * current_price
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cash = self.initial_balance
        self.shares = 0
        self.portfolio_values = [self.initial_balance]
        self.trades = []
        
        obs = self._get_observation()
        info = {
            'cash': self.cash,
            'shares': self.shares,
            'portfolio_value': self.initial_balance
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one trading step
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Store previous portfolio value
        prev_portfolio_value = self._calculate_portfolio_value()
        
        # Execute trade
        transaction_cost, shares_traded = self._execute_trade(action)
        
        # Record trade
        if shares_traded != 0:
            self.trades.append({
                'step': self.current_step,
                'action': action,
                'shares': shares_traded,
                'price': self._get_current_price(),
                'cost': transaction_cost
            })
        
        # Move to next time step
        self.current_step += 1
        
        # Calculate new portfolio value
        current_portfolio_value = self._calculate_portfolio_value()
        self.portfolio_values.append(current_portfolio_value)
        
        # Calculate reward: percentage change in portfolio value (normalized)
        reward = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        # Scale reward to reasonable range (multiply by 100 to get percentage points)
        reward = reward * 100.0
        
        # Check if episode is done
        max_steps = len(self.df) - self.window_size
        terminated = self.current_step >= max_steps - 1
        truncated = False
        
        # Get next observation
        obs = self._get_observation() if not terminated else self._get_observation()
        
        # Info dict
        info = {
            'cash': self.cash,
            'shares': self.shares,
            'portfolio_value': current_portfolio_value,
            'transaction_cost': transaction_cost,
            'shares_traded': shares_traded,
            'price': self._get_current_price()
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render current state (simple text output)"""
        current_price = self._get_current_price()
        portfolio_value = self._calculate_portfolio_value()
        
        print(f"Step: {self.current_step}")
        print(f"Price: ${current_price:.2f}")
        print(f"Shares: {self.shares}")
        print(f"Cash: ${self.cash:.2f}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Return: {(portfolio_value / self.initial_balance - 1) * 100:.2f}%")
        print("-" * 50)
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for the completed episode"""
        portfolio_values = np.array(self.portfolio_values)
        
        # Returns
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        
        # Daily returns
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cumulative_max) / cumulative_max
        max_drawdown = np.min(drawdowns)
        
        # Turnover
        total_traded = sum(abs(trade['shares']) * trade['price'] for trade in self.trades)
        avg_portfolio_value = np.mean(portfolio_values)
        turnover = total_traded / avg_portfolio_value if avg_portfolio_value > 0 else 0
        
        # Profitable days
        profitable_days = np.sum(daily_returns > 0) / len(daily_returns) if len(daily_returns) > 0 else 0
        
        stats = {
            'total_return': total_return,
            'final_value': portfolio_values[-1],
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'turnover': turnover,
            'profitable_days_pct': profitable_days,
            'num_trades': len(self.trades),
            'avg_daily_return': np.mean(daily_returns) if len(daily_returns) > 0 else 0,
            'volatility': np.std(daily_returns) if len(daily_returns) > 0 else 0
        }
        
        return stats
