from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

TRAINING_COLS = ['wallet_age',
    'incoming_tx_count', 'outgoing_tx_count', 'net_incoming_tx_count',
    'total_gas_paid_eth', 'avg_gas_paid_per_tx_eth', 'risky_tx_count',
    'risky_unique_contract_count', 'risky_first_tx_timestamp',
    'risky_last_tx_timestamp', 'risky_first_last_tx_timestamp_diff',
    'risky_sum_outgoing_amount_eth', 'outgoing_tx_sum_eth',
    'incoming_tx_sum_eth', 'outgoing_tx_avg_eth', 'incoming_tx_avg_eth',
    'max_eth_ever', 'min_eth_ever', 'total_balance_eth', 'risk_factor',
    'total_collateral_eth', 'total_collateral_avg_eth',
    'total_available_borrows_eth', 'total_available_borrows_avg_eth',
    'avg_weighted_risk_factor', 'risk_factor_above_threshold_daily_count',
    'avg_risk_factor', 'max_risk_factor', 'borrow_amount_sum_eth',
    'borrow_amount_avg_eth', 'borrow_count', 'repay_amount_sum_eth',
    'repay_amount_avg_eth', 'repay_count', 'borrow_repay_diff_eth',
    'deposit_count', 'deposit_amount_sum_eth', 'time_since_first_deposit',
    'withdraw_amount_sum_eth', 'withdraw_deposit_diff_If_positive_eth',
    'liquidation_count', 'time_since_last_liquidated',
    'liquidation_amount_sum_eth', 'market_adx', 'market_adxr', 'market_apo',
    'market_aroonosc', 'market_aroonup', 'market_atr', 'market_cci',
    'market_cmo', 'market_correl', 'market_dx', 'market_fastk',
    'market_fastd', 'market_ht_trendmode', 'market_linearreg_slope',
    'market_macd_macdext', 'market_macd_macdfix', 'market_macd',
    'market_macdsignal_macdext', 'market_macdsignal_macdfix',
    'market_macdsignal', 'market_max_drawdown_365d', 'market_natr',
    'market_plus_di', 'market_plus_dm', 'market_ppo', 'market_rocp',
    'market_rocr', 'unique_borrow_protocol_count',
    'unique_lending_protocol_count']
    
TARGET_COL = 'target'

DATA_PATH = Path('data/0xFDC1BE05aD924e6Fc4Ab2c6443279fF7C0AB5544_training_data.parquet')

def load_split_scaled_data(test_size=0.2,random_state=24354325) -> tuple[np.array,np.array,np.array,np.array]:
    df = pd.read_parquet(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(df[TRAINING_COLS].to_numpy(),df[TARGET_COL].to_numpy(),
                                                        test_size=test_size,random_state=random_state)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test
