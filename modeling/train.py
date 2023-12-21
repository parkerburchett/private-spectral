from model import PredictLiquidationsV1, device
from their_modeling import TestData,TrainData
from helpers import load_split_scaled_data
from torch.utils.data import DataLoader
import torch


def main():
    X_train_scaled, X_val_scaled, y_train, y_val = load_split_scaled_data()

    model = PredictLiquidationsV1(input_features=X_train_scaled.shape[1],
                                  output_features=1,
                                  hidden_units=100).to(device)
    
    train_data = TrainData(torch.from_numpy(X_train_scaled).type(torch.float),
                       torch.from_numpy(y_train).type(torch.float))

    validation_data = TestData(torch.from_numpy(X_val_scaled).type(torch.float),
                        torch.from_numpy(y_val).type(torch.float))
    
    pass


if __name__ == "__main__":
    main()
