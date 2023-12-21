# %%
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_parquet('data/0xFDC1BE05aD924e6Fc4Ab2c6443279fF7C0AB5544_training_data.parquet')
test_df = df[-10_000:]
train_df = df[:-10_000]
device = 'cuda'

training_cols = ['wallet_age',
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
       'unique_lending_protocol_count',]

target_cols = 'target'

X_train, X_test, y_train, y_test = train_test_split(train_df[training_cols].to_numpy(),
                                                    train_df['target'].to_numpy(),
                                                    test_size=0.2,
                                                    random_state=24354325)


# %%
import torch
from torch import nn
from torch import optim
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

input_dim = len(training_cols)

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="relu")
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="relu")
        self.layer_4 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_4.weight, nonlinearity="relu")
        self.layer_5 = nn.Linear(hidden_dim, output_dim)
       
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = torch.nn.functional.relu(self.layer_3(x))
        x = torch.nn.functional.relu(self.layer_4(x))
        x = torch.nn.functional.sigmoid(self.layer_5(x))
        return x
    

def setup_model(hidden_dim:int):
    model = NeuralNetwork(len(training_cols), hidden_dim, 1)
    model.to(device)    
    return model


class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len


def make_dataloaders(batch_size):
    # Instantiate training and test data
    train_data = Data(X_train, y_train)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = Data(X_test, y_test)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def train_model(model, train_dataloader, test_dataloader, num_epochs, learning_rate):
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        loss_values = []
        for X, y in train_dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y.unsqueeze(-1))
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()


             # Print gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Epoch: {epoch}, Layer: {name}, Gradient norm: {param.grad.norm().item()}")
        
        print(np.mean(loss_values).round(5), 'training loss', epoch, 'epoch')

num_epochs = 100
learning_rate = 0.01
hidden_dim = 50
model = setup_model(hidden_dim)
train_dataloader, test_dataloader = make_dataloaders(2**12)

train_model(model, train_dataloader, test_dataloader, num_epochs, learning_rate)


# %%



