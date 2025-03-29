import os
import numpy as np
import torch
from torch.utils.data import Dataset

class UCIHAR_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_uci_har_data(data_dir):
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_", "body_acc_y_", "body_acc_z_",
        "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
        "total_acc_x_", "total_acc_y_", "total_acc_z_"
    ]

    def load_signals(subset):
        signals_data = []
        for signal in INPUT_SIGNAL_TYPES:
            filepath = os.path.join(data_dir, subset, "Inertial Signals", signal + subset + ".txt")
            signals = np.loadtxt(filepath)
            signals_data.append(signals)
        return np.transpose(np.array(signals_data), (1, 0, 2))

    def load_labels(subset):
        filepath = os.path.join(data_dir, subset, "y_" + subset + ".txt")
        return np.loadtxt(filepath) - 1

    X_train = load_signals("train")
    y_train = load_labels("train")
    X_test = load_signals("test")
    y_test = load_labels("test")
    return X_train, y_train, X_test, y_test
