import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (T, 1, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (T, B, d_model)
        """
        return x + self.pe[:x.size(0)]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = (
            nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride=stride), nn.BatchNorm1d(out_channels))
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetTransformer(nn.Module):
    """"
    [body_acc_x, body_acc_y, body_acc_z,
     body_gyro_x, body_gyro_y, body_gyro_z,
     total_acc_x, total_acc_y, total_acc_z]  ← 길이 9 짜리 벡터
     """
    def __init__(self, num_classes, input_channels=9, seq_len=128, d_model=64, nhead=4, num_layers=1):
        super().__init__()

        # Stage 1
        self.stage1_resnet = nn.Sequential(
            ResidualBlock(input_channels, 32),
            ResidualBlock(32, 64)
        )
        self.stage1_proj = nn.Linear(seq_len, d_model)
        self.pos_encoder1 = PositionalEncoding(d_model, max_len=64)
        encoder_layer1 = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1)
        self.stage1_transformer = TransformerEncoder(encoder_layer1, num_layers=num_layers)

        # Stage 2
        self.stage2_resnet = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.stage2_proj = nn.Linear(d_model, d_model)
        self.pos_encoder2 = PositionalEncoding(d_model, max_len=64)
        encoder_layer2 = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1)
        self.stage2_transformer = TransformerEncoder(encoder_layer2, num_layers=num_layers)

        self.fc = nn.Linear(d_model * 64, num_classes)

    def forward(self, x):
        # Stage 1
        x = self.stage1_resnet(x)                   # (B, 64, T)
        x = self.stage1_proj(x)                     # (B, 64, d_model)
        x = x.permute(2, 0, 1)                      # (T=64, B, d_model)
        x = self.pos_encoder1(x)                    # Positional Encoding
        x = self.stage1_transformer(x)              # Transformer
        x = x.permute(1, 2, 0)                      # (B, d_model, T)
        x = x.transpose(1, 2)                       # (B, T, d_model)

        # Stage 2
        x = self.stage2_resnet(x)                   # (B, 64, d_model)
        x = self.stage2_proj(x)                     # (B, 64, d_model)
        x = x.permute(2, 0, 1)                      # (T=64, B, d_model)
        x = self.pos_encoder2(x)                    # Positional Encoding
        x = self.stage2_transformer(x)              # Transformer
        x = x.permute(1, 2, 0)                      # (B, 64, d_model)

        x = x.flatten(start_dim=1)                  # (B, 64*d_model)
        return self.fc(x)
