import torch.nn as nn

class PriceTransformer(nn.Module):
    """
    Extremely simple Transformer model for price forecasting.
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(PriceTransformer, self).__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Output is currently unbounded, but could be in the range of [0, 1] to improve performance.
        """
        # x shape: (batch_size, seq_len, feature_dim)
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        x = x[:, -1, :]  # use the last time step's output
        out = self.output_proj(x)  # (batch_size, 1)
        return out.squeeze(-1)  # (batch_size,) - continuous price values