import torch
import pandas as pd
if torch.cuda.is_available():
    device = 'cuda'
else:
    raise ValueError("not on cuda")


def main():
    df = pd.read_parquet('data/0xFDC1BE05aD924e6Fc4Ab2c6443279fF7C0AB5544_training_data.parquet')
    test_df = df[-10_000:]
    train_df = df[:-10_000]
    



if __name__ == '__main__':
    main()