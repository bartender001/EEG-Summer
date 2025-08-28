import torch
import torch.nn as nn
import torch.nn.functional as F

class EEG_BiGRU_CNN(nn.Module):
    def __init__(self, input_features=44, num_classes=4, gru_units=192,
                 cnn_filters=64, cnn_kernel=5, dropout=0.3):
        super(EEG_BiGRU_CNN, self).__init__()

        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.bn1 = nn.BatchNorm1d(gru_units * 2)
        self.dropout_spatial = nn.Dropout2d(0.15)

        self.conv1d = nn.Conv1d(
            in_channels=gru_units * 2,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel,
            padding='same'
        )
        self.bn2 = nn.BatchNorm1d(cnn_filters)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(cnn_filters * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)

        x = gru_out.permute(0, 2, 1)
        x = self.bn1(x)
        x = self.dropout_spatial(x.unsqueeze(3)).squeeze(3)

        x = F.relu(self.conv1d(x))
        x = self.bn2(x)

        avg_pool_out = self.avg_pool(x)
        max_pool_out = self.max_pool(x)

        avg_pool_out = avg_pool_out.squeeze(-1)
        max_pool_out = max_pool_out.squeeze(-1)

        pooled_out = torch.cat([avg_pool_out, max_pool_out], dim=1)

        out = self.classifier(pooled_out)
        return out


