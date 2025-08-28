import torch
import torch.nn as nn
import torch.nn.functional as F

class EEG_BiGRU_CNN(nn.Module):
    """
    A hybrid Bidirectional GRU-CNN model for EEG signal classification, inspired
    by the "Fast_BiGRU_CNN" Keras architecture.

    This model first processes the input sequence with a Bidirectional GRU to
    capture temporal dependencies. Then, a lightweight 1D CNN block extracts
    features from the GRU's output sequence. Global pooling summarizes these
    features before they are passed to a final dense classifier.

    NOTE: This model expects preprocessed input where each sample is a window
    of combined time-domain and frequency-domain features.
    The expected input shape is (Batch, Timesteps, Features).

    Args:
        input_features (int): The number of features per time step.
                              (e.g., 44 from 22 channels * 2 feature types).
        num_classes (int): The number of classes for classification.
        gru_units (int): The number of units in the GRU layer.
        cnn_filters (int): The number of filters in the Conv1D layer.
        cnn_kernel (int): The kernel size for the Conv1D layer.
        dropout (float): The dropout rate.
    """
    def __init__(self, input_features=44, num_classes=4, gru_units=192,
                 cnn_filters=64, cnn_kernel=5, dropout=0.3):
        super(EEG_BiGRU_CNN, self).__init__()

        # --- Bidirectional GRU Layer ---
        # Processes the sequence of input features.
        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # We need to apply BatchNorm to the feature dimension.
        # GRU output is (B, T, F), so BatchNorm1d expects (B, F, T).
        self.bn1 = nn.BatchNorm1d(gru_units * 2) # *2 for bidirectional
        self.dropout_spatial = nn.Dropout2d(0.15) # Simulates SpatialDropout1D

        # --- Lightweight CNN Block ---
        self.conv1d = nn.Conv1d(
            in_channels=gru_units * 2,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel,
            padding='same'
        )
        self.bn2 = nn.BatchNorm1d(cnn_filters)

        # --- Global Pooling ---
        # These will operate on the time dimension.
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # --- Dense Classifier ---
        # Input size is 2 * cnn_filters because we concatenate avg and max pooling.
        self.classifier = nn.Sequential(
            nn.Linear(cnn_filters * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the network.
        Expected input shape: (Batch, Timesteps, Features) -> e.g., (32, 50, 44)
        """
        # 1. Pass through GRU
        # x shape: (B, T, F) -> gru_out shape: (B, T, 2*H)
        gru_out, _ = self.gru(x)

        # 2. Apply BatchNorm and Spatial Dropout
        # Permute to (B, F, T) for batch normalization
        x = gru_out.permute(0, 2, 1)
        x = self.bn1(x)
        # Permute to (B, F, T, 1) for spatial dropout, then squeeze back
        x = self.dropout_spatial(x.unsqueeze(3)).squeeze(3)

        # 3. Pass through CNN block
        x = F.relu(self.conv1d(x))
        x = self.bn2(x)

        # 4. Apply Global Pooling
        # avg_pool_out and max_pool_out shape: (B, F_cnn, 1)
        avg_pool_out = self.avg_pool(x)
        max_pool_out = self.max_pool(x)

        # Squeeze the last dimension to get (B, F_cnn)
        avg_pool_out = avg_pool_out.squeeze(-1)
        max_pool_out = max_pool_out.squeeze(-1)

        # 5. Concatenate pooling results
        # pooled_out shape: (B, 2 * F_cnn)
        pooled_out = torch.cat([avg_pool_out, max_pool_out], dim=1)

        # 6. Pass through the final classifier
        out = self.classifier(pooled_out)
        return out

# --- Example Usage ---
if __name__ == '__main__':
    # These dimensions match the successful Keras experiment
    batch_size = 32
    timesteps = 50
    features = 44
    num_classes = 4

    model = EEG_BiGRU_CNN(input_features=features, num_classes=num_classes)
    print("Model Architecture:")
    print(model)

    # Create a dummy input tensor with the correct shape
    dummy_data = torch.randn(batch_size, timesteps, features)
    output = model(dummy_data)

    print(f"\nShape of dummy input data: {dummy_data.shape}")
    print(f"Shape of model output: {output.shape}")
    assert output.shape == (batch_size, num_classes)
    print("\nModel forward pass successful!")

