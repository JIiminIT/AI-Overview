import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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
    def __init__(self, num_classes, input_channels=9, seq_len=128, d_model=64, nhead=4, num_layers=1):
        super().__init__()

        # Stage 1: ResNet + Transformer
        self.stage1_resnet = nn.Sequential(
            ResidualBlock(input_channels, 32),
            ResidualBlock(32, 64)
        )
        self.stage1_proj = nn.Linear(seq_len, d_model)
        encoder_layer1 = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1)
        self.stage1_transformer = TransformerEncoder(encoder_layer1, num_layers=num_layers)

        # Stage 2: ResNet + Transformer
        self.stage2_resnet = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.stage2_proj = nn.Linear(d_model, d_model)  # d_model 유지
        encoder_layer2 = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1)
        self.stage2_transformer = TransformerEncoder(encoder_layer2, num_layers=num_layers)

        self.fc = nn.Linear(d_model * 64, num_classes)

    def forward(self, x):
        # Stage 1
        x = self.stage1_resnet(x)                   # (B, 64, T)
        x = self.stage1_proj(x)                     # (B, 64, d_model)
        x = x.permute(2, 0, 1)                      # (d_model, B, 64)
        x = self.stage1_transformer(x)              # (d_model, B, 64)
        x = x.permute(1, 2, 0)                      # (B, 64, d_model)
        x = x.transpose(1, 2)                       # (B, d_model, 64)

        # Stage 2
        x = self.stage2_resnet(x)                   # (B, 64, d_model)
        x = self.stage2_proj(x)                     # (B, 64, d_model)
        x = x.permute(2, 0, 1)                      # (d_model, B, 64)
        x = self.stage2_transformer(x)              # (d_model, B, 64)
        x = x.permute(1, 2, 0)                      # (B, 64, d_model)

        x = x.flatten(start_dim=1)             # (B, 64*d_model)
        return self.fc(x)
