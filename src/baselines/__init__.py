"""Baselines package"""
from .baseline_strategies import (
    BuyAndHold,
    RandomPolicy,
    MovingAverageCrossover,
    MomentumStrategy,
    RSIStrategy,
    evaluate_all_baselines
)

__all__ = [
    'BuyAndHold',
    'RandomPolicy', 
    'MovingAverageCrossover',
    'MomentumStrategy',
    'RSIStrategy',
    'evaluate_all_baselines'
]
