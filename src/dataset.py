import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler

class EEGDataset(Dataset):
    def __init__(self, csv_path, epoch_len=201, window_size=50, stride=10):
        df = pd.read_csv(csv_path)
        df = df.dropna()

        eeg_cols = [c for c in df.columns if c.startswith("EEG-")]
        
        grouped = df.groupby('epoch')
        X_epochs, y_epochs = [], []
        for _, group in grouped:
            data = group[eeg_cols].values.T.astype(np.float32)
            
            if data.shape[1] > epoch_len:
                data = data[:, :epoch_len]
            elif data.shape[1] < epoch_len:
                pad_width = epoch_len - data.shape[1]
                data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
            
            X_epochs.append(data)
            y_epochs.append(group['label'].iloc[0])

        X_epochs = np.array(X_epochs)
        y_epochs = np.array(y_epochs)

        n_samples, n_channels, n_time_points = X_epochs.shape

        X_normalized = np.zeros_like(X_epochs)
        for i in range(n_samples):
            for ch in range(n_channels):
                scaler = StandardScaler()
                X_normalized[i, ch, :] = scaler.fit_transform(X_epochs[i, ch, :].reshape(-1, 1)).flatten()

        X_freq = np.abs(np.fft.rfft(X_normalized, axis=-1))
        
        X_windows, y_windows = [], []
        num_windows_per_epoch = (n_time_points - window_size) // stride + 1

        for i in range(n_samples):
            for w in range(num_windows_per_epoch):
                start = w * stride
                end = start + window_size

                window_time = X_normalized[i, :, start:end]

                freq_len = X_freq.shape[2]
                freq_start = (start * freq_len) // n_time_points
                freq_end = (end * freq_len) // n_time_points
                window_freq_raw = X_freq[i, :, freq_start:freq_end]
                
                if window_freq_raw.shape[1] < window_size:
                    pad_w = window_size - window_freq_raw.shape[1]
                    window_freq = np.pad(window_freq_raw, ((0,0), (0,pad_w)), 'constant')
                else:
                    window_freq = window_freq_raw[:, :window_size]

                combined_window = np.stack([window_time, window_freq], axis=-1)
                
                combined_window = combined_window.transpose(1, 0, 2)
                final_window = combined_window.reshape(window_size, -1)
                
                X_windows.append(final_window)
                y_windows.append(y_epochs[i])

        self.X = np.array(X_windows, dtype=np.float32)
        self.y_str = np.array(y_windows)

        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y_str)
        self.num_classes = len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return x_tensor, y_tensor

