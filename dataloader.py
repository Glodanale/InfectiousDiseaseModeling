# dataloader.py

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


class SEIRDDataLoader:
    def __init__(self, dataset_path='../../SEIR_CSV.csv', sequence_length=50, scaler_path='scaler.pkl'):
        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        self.scaler_path = scaler_path
        self.scaler = self._load_or_create_scaler()
        self.data = self._load_and_scale_data()
        self.num_features = self.data.shape[1]

    def _load_or_create_scaler(self):
        try:
            with open(self.scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("✅ Scaler loaded successfully!")
        except (FileNotFoundError, EOFError):
            scaler = MinMaxScaler()
            print("⚠️ Scaler not found. Creating a new one.")
        return scaler

    def _save_scaler(self):
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print("✅ Scaler saved successfully!")

    def _load_and_scale_data(self):
        df = pd.read_csv(self.dataset_path).astype(float)
        scaled_data = self.scaler.fit_transform(df)
        self._save_scaler()
        return scaled_data

    def create_sequences(self):
        X, y = [], []
        for i in range(len(self.data) - self.sequence_length):
            X.append(self.data[i:i + self.sequence_length])
            y.append(self.data[i + self.sequence_length])
        return np.array(X), np.array(y)


# ------------------ SPLIT STRATEGIES ------------------ #

def validation_split(X, y, val_size=0.2, test_size=0.2, batch_size=32, shuffle=False, random_state=None):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, shuffle=shuffle, random_state=random_state
    )
    return _make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)


def walkforward_split(X, y, window_size, horizon, batch_size=32):
    splits = []
    start = 0

    while start + window_size + horizon <= len(X):
        # Expanding training window from 0 to start + window_size
        end_train = start + window_size
        end_val = end_train + horizon

        X_train = X[0:end_train]
        y_train = y[0:end_train]

        X_val = X[end_train:end_val]
        y_val = y[end_train:end_val]

        # Use validation set as test loader for evaluation
        loaders = _make_loaders(X_train, y_train, X_val, y_val, X_val, y_val, batch_size)
        splits.append(loaders)

        start += horizon  # Move forward for next fold

    return splits




def rolling_split(X, y, window_size, stride, batch_size=32):
    splits = []
    for start in range(0, len(X) - window_size - stride + 1, stride):
        end_train = start + window_size
        end_val = end_train + stride
        X_train, y_train = X[start:end_train], y[start:end_train]
        X_val, y_val = X[end_train:end_val], y[end_train:end_val]
        splits.append(_make_loaders(X_train, y_train, X_val, y_val, None, None, batch_size))
    return splits


def to_loader(X, y, batch_size, shuffle=False, drop_last=False):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def _make_loaders(X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, batch_size=32):
    train_loader = to_loader(X_train, y_train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = to_loader(X_val, y_val, batch_size=batch_size, drop_last=False) if X_val is not None else None
    test_loader = to_loader(X_test, y_test, batch_size=batch_size, drop_last=False) if X_test is not None else None

    return train_loader, val_loader, test_loader



# ------------------ DISPATCH WRAPPER ------------------ #

def get_split_strategy(data, strategy, sequence_length, batch_size, test_size=0.2, val_size=0.2, **kwargs):
    X, y = create_sequences_from_data(data, sequence_length)

    if strategy == "train_val_test":
        return validation_split(X, y, val_size, test_size, batch_size)

    elif strategy == "walkforward":
        return walkforward_split(
            X, y, window_size=kwargs.get("window_size", 200),
            horizon=kwargs.get("horizon", 25), batch_size=batch_size
        )

    elif strategy == "rolling":
        return rolling_split(
            X, y, window_size=kwargs.get("window_size", 200),
            stride=kwargs.get("stride", 25), batch_size=batch_size
        )

    else:
        raise ValueError(f"Unknown split strategy: {strategy}")


def create_sequences_from_data(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)
