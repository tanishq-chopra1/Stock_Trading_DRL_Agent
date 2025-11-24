"""
Evaluation metrics for trading strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def calculate_returns(portfolio_values: np.ndarray) -> float:
    """Calculate total return"""
    return (portfolio_values[-1] / portfolio_values[0]) - 1


def calculate_cagr(portfolio_values: np.ndarray, n_years: float) -> float:
    """
    Calculate Compound Annual Growth Rate
    
    Args:
        portfolio_values: Array of portfolio values
        n_years: Number of years
    """
    total_return = portfolio_values[-1] / portfolio_values[0]
    cagr = (total_return ** (1 / n_years)) - 1
    return cagr


def calculate_sharpe_ratio(
    returns: np.ndarray, 
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily)
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if np.std(returns) == 0:
        return 0.0
    sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(periods_per_year)
    return sharpe


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation instead of total volatility)
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(periods_per_year)
    return sortino


def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """
    Calculate maximum drawdown
    
    Returns:
        max_drawdown: Maximum drawdown as a negative percentage
    """
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - cumulative_max) / cumulative_max
    max_drawdown = np.min(drawdowns)
    return max_drawdown


def calculate_calmar_ratio(
    portfolio_values: np.ndarray,
    n_years: float
) -> float:
    """
    Calculate Calmar ratio (CAGR / abs(max drawdown))
    """
    cagr = calculate_cagr(portfolio_values, n_years)
    max_dd = abs(calculate_max_drawdown(portfolio_values))
    
    if max_dd == 0:
        return 0.0
    
    return cagr / max_dd


def calculate_win_rate(returns: np.ndarray) -> float:
    """Calculate percentage of profitable periods"""
    return np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0.0


def calculate_profit_factor(returns: np.ndarray) -> float:
    """
    Calculate profit factor (total gains / total losses)
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    
    return gains / losses


def calculate_all_metrics(
    portfolio_values: np.ndarray,
    n_trading_days: int = 252
) -> Dict[str, float]:
    """
    Calculate all trading metrics
    
    Args:
        portfolio_values: Array of portfolio values over time
        n_trading_days: Number of trading days per year
        
    Returns:
        Dictionary of all metrics
    """
    # Calculate returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    total_return = calculate_returns(portfolio_values)
    
    # Calculate time period
    n_years = len(portfolio_values) / n_trading_days
    
    # Calculate all metrics
    metrics = {
        'total_return': total_return,
        'cagr': calculate_cagr(portfolio_values, n_years) if n_years > 0 else 0.0,
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'max_drawdown': calculate_max_drawdown(portfolio_values),
        'calmar_ratio': calculate_calmar_ratio(portfolio_values, n_years) if n_years > 0 else 0.0,
        'volatility': np.std(returns) * np.sqrt(n_trading_days),
        'win_rate': calculate_win_rate(returns),
        'profit_factor': calculate_profit_factor(returns),
        'final_value': portfolio_values[-1],
        'n_periods': len(portfolio_values),
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Performance Metrics"):
    """Pretty print metrics"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    print(f"Total Return:        {metrics['total_return']:>10.2%}")
    print(f"CAGR:                {metrics['cagr']:>10.2%}")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
    print(f"Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
    print(f"Max Drawdown:        {metrics['max_drawdown']:>10.2%}")
    print(f"Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")
    print(f"Volatility (Ann.):   {metrics['volatility']:>10.2%}")
    print(f"Win Rate:            {metrics['win_rate']:>10.2%}")
    print(f"Profit Factor:       {metrics['profit_factor']:>10.2f}")
    print(f"Final Value:         ${metrics['final_value']:>10,.2f}")
    print(f"{'='*60}\n")


def compare_strategies(
    results: Dict[str, np.ndarray],
    strategy_names: List[str] = None
) -> pd.DataFrame:
    """
    Compare multiple strategies
    
    Args:
        results: Dict mapping strategy name to portfolio values array
        strategy_names: Optional list of strategy names to include
        
    Returns:
        DataFrame with metrics for each strategy
    """
    if strategy_names is None:
        strategy_names = list(results.keys())
    
    comparison_data = []
    
    for name in strategy_names:
        if name in results:
            metrics = calculate_all_metrics(results[name])
            metrics['strategy'] = name
            comparison_data.append(metrics)
    
    df = pd.DataFrame(comparison_data)
    df = df.set_index('strategy')
    
    # Reorder columns for better readability
    column_order = [
        'total_return', 'cagr', 'sharpe_ratio', 'sortino_ratio',
        'max_drawdown', 'calmar_ratio', 'volatility', 
        'win_rate', 'profit_factor', 'final_value', 'n_periods'
    ]
    df = df[[col for col in column_order if col in df.columns]]
    
    return df
