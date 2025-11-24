"""Download and check pre-COVID data split"""
from src.utils.data_utils import prepare_data_for_trading

train, val, test = prepare_data_for_trading(
    'SPY', 
    '2010-01-01', 
    '2020-01-01', 
    data_dir='data', 
    add_indicators=False
)

print("\nPre-COVID data successfully downloaded!")
print("This avoids the distribution shift from COVID crash/recovery")
