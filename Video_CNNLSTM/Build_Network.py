import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


class C3DModule(nn.Module):
    def __init__(self, downsample_size=(4, 4)):
        super(C3DModule, self).__init__()
        # 初始下采样

        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.pool3 = nn.MaxPool3d((1, 2, 2))
        self.conv4 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.pool4 = nn.AdaptiveMaxPool3d((None, 1, 1))

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        # Flatten the last two dimensions
        x = x.view(x.size(0), x.size(1), -1)  # Change shape to (batch_size, seq_length, features)
        return x
class TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        return F.relu(self.dropout1(out))

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, dilation=dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.downsample = nn.Conv1d(num_inputs, num_channels[-1], 1) if num_inputs != num_channels[-1] else None

    def forward(self, x):
        # x needs to be (batch_size, channels, seq_length)
        out = self.tcn(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class TCN_LSTM(nn.Module):
    def __init__(self, num_channels=[16, 32], hidden_size=128, kernel_size=1, dropout=0.2, num_layers=1, bidirectional=False):
        super(TCN_LSTM, self).__init__()
        self.tcn = TCN(128, num_channels, kernel_size, dropout)  # 3 input channels for video frames
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.c3d= C3DModule()

    def forward(self, x):
        #(batch_size, seq_length, H, W,channels)

        # Assuming x is of shape (batch_size, seq_length, channels, H, W)
        x=x.permute(0, 4, 1, 2, 3)
        #(batch_size, seq_length, channels, H, W)
        x = self.c3d(x)
        # Permute to match TCN input requirements (batch_size, channels, seq_length)
        x = self.tcn(x)
        x=x.permute(0, 2, 1)

        # LSTM expects (batch_size, seq_length, features), no permute needed after TCN
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])  # Use only the last output from LSTM
        return self.fc(x)