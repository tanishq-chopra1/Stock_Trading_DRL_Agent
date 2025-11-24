"""
Visualization utilities for trading analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def plot_portfolio_values(
    results: Dict[str, np.ndarray],
    title: str = "Portfolio Value Comparison",
    save_path: str = None,
    normalize: bool = True
):
    """
    Plot portfolio values over time for multiple strategies
    
    Args:
        results: Dict mapping strategy name to portfolio values
        title: Plot title
        save_path: Path to save figure
        normalize: Whether to normalize to 100
    """
    plt.figure(figsize=(14, 8))
    
    for name, values in results.items():
        if normalize:
            plot_values = (values / values[0]) * 100
            ylabel = 'Portfolio Value (Normalized to 100)'
        else:
            plot_values = values
            ylabel = 'Portfolio Value ($)'
        
        plt.plot(plot_values, label=name, linewidth=2, alpha=0.8)
    
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_drawdowns(
    results: Dict[str, np.ndarray],
    title: str = "Drawdown Comparison",
    save_path: str = None
):
    """Plot drawdowns over time"""
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 4*len(results)), sharex=True)
    
    if len(results) == 1:
        axes = [axes]
    
    for (name, values), ax in zip(results.items(), axes):
        cumulative_max = np.maximum.accumulate(values)
        drawdowns = (values - cumulative_max) / cumulative_max * 100
        
        ax.fill_between(range(len(drawdowns)), drawdowns, 0, alpha=0.3, color='red')
        ax.plot(drawdowns, color='red', linewidth=1)
        ax.set_ylabel('Drawdown (%)', fontsize=10)
        ax.set_title(f'{name} - Max DD: {np.min(drawdowns):.2f}%', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    axes[-1].set_xlabel('Trading Days', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_returns_distribution(
    results: Dict[str, np.ndarray],
    title: str = "Daily Returns Distribution",
    save_path: str = None
):
    """Plot distribution of daily returns"""
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5), sharey=True)
    
    if len(results) == 1:
        axes = [axes]
    
    for (name, values), ax in zip(results.items(), axes):
        returns = np.diff(values) / values[:-1] * 100  # Percentage returns
        
        ax.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=np.mean(returns), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(returns):.3f}%')
        ax.set_xlabel('Daily Return (%)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(name, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_metrics_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str] = None,
    title: str = "Metrics Comparison",
    save_path: str = None
):
    """
    Plot bar charts comparing metrics across strategies
    
    Args:
        comparison_df: DataFrame from compare_strategies()
        metrics: List of metrics to plot
        title: Plot title
        save_path: Path to save figure
    """
    if metrics is None:
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for metric, ax in zip(metrics, axes):
        if metric in comparison_df.columns:
            data = comparison_df[metric].sort_values(ascending=False)
            
            # Color bars
            colors = ['green' if x > 0 else 'red' for x in data.values]
            if metric == 'max_drawdown':
                colors = ['red' if x < 0 else 'green' for x in data.values]
            
            data.plot(kind='bar', ax=ax, color=colors, alpha=0.7, edgecolor='black')
            ax.set_title(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_xlabel('Strategy', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Format y-axis
            if metric in ['total_return', 'max_drawdown', 'win_rate']:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    plt.xticks(rotation=45, ha='right')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_training_curves(
    log_file: str,
    metrics: List[str] = ['total_return', 'sharpe_ratio'],
    title: str = "Training Curves",
    save_path: str = None
):
    """
    Plot training and validation curves over episodes
    
    Args:
        log_file: Path to training log CSV
        metrics: Metrics to plot
        title: Plot title
        save_path: Path to save figure
    """
    # Load log
    df = pd.read_csv(log_file)
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics), sharex=True)
    
    if n_metrics == 1:
        axes = [axes]
    
    for metric, ax in zip(metrics, axes):
        if f'train_{metric}' in df.columns and f'val_{metric}' in df.columns:
            ax.plot(df['episode'], df[f'train_{metric}'], 
                   label='Train', linewidth=2, alpha=0.7)
            ax.plot(df['episode'], df[f'val_{metric}'], 
                   label='Validation', linewidth=2, alpha=0.7)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Episode', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def create_summary_report(
    results: Dict[str, np.ndarray],
    comparison_df: pd.DataFrame,
    output_dir: str = 'results'
):
    """
    Create a comprehensive visual report
    
    Args:
        results: Portfolio values for each strategy
        comparison_df: Metrics comparison DataFrame
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating comprehensive report...")
    
    # 1. Portfolio values
    plot_portfolio_values(
        results,
        title="Portfolio Performance Comparison",
        save_path=output_path / "portfolio_values.png"
    )
    
    # 2. Drawdowns
    plot_drawdowns(
        results,
        title="Drawdown Analysis",
        save_path=output_path / "drawdowns.png"
    )
    
    # 3. Returns distribution
    plot_returns_distribution(
        results,
        title="Daily Returns Distribution",
        save_path=output_path / "returns_distribution.png"
    )
    
    # 4. Metrics comparison
    plot_metrics_comparison(
        comparison_df,
        metrics=['total_return', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio'],
        title="Performance Metrics Comparison",
        save_path=output_path / "metrics_comparison.png"
    )
    
    print(f"\nReport generated in {output_dir}/")
    print("Files created:")
    print("  - portfolio_values.png")
    print("  - drawdowns.png")
    print("  - returns_distribution.png")
    print("  - metrics_comparison.png")
