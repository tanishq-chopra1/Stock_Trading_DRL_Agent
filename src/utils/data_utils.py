"""
Data downloading and preprocessing utilities
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import argparse


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    data_dir: str = 'data'
) -> pd.DataFrame:
    """
    Download stock data using yfinance and cache locally
    
    Args:
        ticker: Stock ticker symbol (e.g., 'SPY', 'AAPL')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_dir: Directory to save cached data
        
    Returns:
        DataFrame with OHLCV data
    """
    # Create data directory
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if data already cached
    cache_file = Path(data_dir) / f"{ticker}_{start_date}_{end_date}.csv"
    
    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Check if cached file has corrupted column names from MultiIndex
        # They would look like "('SPY', 'Close')" as strings
        if any('(' in str(col) and ')' in str(col) for col in df.columns):
            print("Cached file has corrupted columns, re-downloading...")
            cache_file.unlink()  # Delete corrupted cache
            df = yf.download(ticker, start=start_date, end=end_date, progress=True)
            
            # Standardize column names before saving
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [col.lower() for col in df.columns]
            
            # Save to cache with clean columns
            df.to_csv(cache_file)
            print(f"Data saved to {cache_file}")
        else:
            # Cached file is clean, just standardize column names
            df.columns = [col.lower() for col in df.columns]
    else:
        print(f"Downloading {ticker} data from {start_date} to {end_date}...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=True)
        
        # Standardize column names (handle multi-index from yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [col.lower() for col in df.columns]
        
        # Save to cache
        df.to_csv(cache_file)
        print(f"Data saved to {cache_file}")
    
    # Remove any NaN rows
    df = df.dropna()
    
    print(f"Loaded {len(df)} trading days")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional technical indicator columns
    """
    df = df.copy()
    
    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Moving averages
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # Exponential moving averages
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_30'] = df['close'].ewm(span=30, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    df['volatility_30'] = df['returns'].rolling(window=30).std()
    
    # Volume indicators
    df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_10']
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # Remove NaN rows created by indicators
    df = df.dropna()
    
    return df


def create_train_val_test_split(
    df: pd.DataFrame,
    train_pct: float = 0.7,
    val_pct: float = 0.15,
    test_pct: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train, validation, and test sets
    
    Args:
        df: Input dataframe
        train_pct: Percentage for training (e.g., 0.7 = 70%)
        val_pct: Percentage for validation
        test_pct: Percentage for testing
        
    Returns:
        train_df, val_df, test_df
    """
    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-6, "Percentages must sum to 1.0"
    
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"\nData split:")
    print(f"Train: {len(train_df)} samples ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"Val:   {len(val_df)} samples ({val_df.index[0]} to {val_df.index[-1]})")
    print(f"Test:  {len(test_df)} samples ({test_df.index[0]} to {test_df.index[-1]})")
    
    return train_df, val_df, test_df


def prepare_data_for_trading(
    ticker: str,
    start_date: str,
    end_date: str,
    data_dir: str = 'data',
    add_indicators: bool = True,
    train_pct: float = 0.7,
    val_pct: float = 0.15,
    test_pct: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete data preparation pipeline
    
    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date
        data_dir: Data directory
        add_indicators: Whether to add technical indicators
        train_pct: Training set percentage
        val_pct: Validation set percentage
        test_pct: Test set percentage
        
    Returns:
        train_df, val_df, test_df
    """
    # Download data
    df = download_stock_data(ticker, start_date, end_date, data_dir)
    
    # Add technical indicators
    if add_indicators:
        print("\nAdding technical indicators...")
        df = add_technical_indicators(df)
    
    # Split data
    train_df, val_df, test_df = create_train_val_test_split(
        df, train_pct, val_pct, test_pct
    )
    
    return train_df, val_df, test_df


def main():
    """Command-line interface for downloading data"""
    parser = argparse.ArgumentParser(description='Download stock data for RL trading')
    parser.add_argument('--ticker', type=str, default='SPY', help='Stock ticker symbol')
    parser.add_argument('--start', type=str, default='2014-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--add_indicators', action='store_true', help='Add technical indicators')
    
    args = parser.parse_args()
    
    # Download and prepare data
    train_df, val_df, test_df = prepare_data_for_trading(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        data_dir=args.data_dir,
        add_indicators=args.add_indicators
    )
    
    print("\nData preparation complete!")
    print(f"Available features: {list(train_df.columns)}")


if __name__ == '__main__':
    main()
