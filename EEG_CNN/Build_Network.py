import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

class SeizureDetectionCNN(nn.Module):
    def __init__(self):
        super(SeizureDetectionCNN, self).__init__()
        # 由于输入的数据形状是 [batch_size, length, channels]，且channels=21，
        # 我们需要调整第一个卷积层的in_channels为21。
        self.conv1 = nn.Conv1d(21, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 100)
        self.fc2 =nn.Linear(100, 1)# 假设你有100个类别进行分类

    def forward(self, x):
        # 在forward函数中，你需要确保数据是按照正确的维度进入各层
        x = x.transpose(1, 2)

        # Proceed with the model operations
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.bn2(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x= self.fc2(x)
        return x

# 实例化模型




class EEGHybridNet(nn.Module):
    def __init__(self):
        super(EEGHybridNet, self).__init__()
        # Assuming the concatenated feature size from three domains
        self.cnn = nn.Conv1d(78, 64, kernel_size=3, padding=1)  # Adjust in_channels based on your data concatenation
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)  # Output layer for binary classification

    def forward(self, x):
        # x shape: (batch_size=256, channels=21, timesteps=512)

        # Time-domain data (no change)
        time_data = x

        # Frequency-domain data (FFT)
        # Note: Adjust the fft computation according to the pytorch version and make sure dimensions are correct.
        freq_data = torch.fft.rfft(x, dim=2)
        freq_data = torch.abs(freq_data)

        # Time-frequency domain data (DWT)
        # Handling with batch processing for DWT
        time_freq_data = []
        for sample in x:
            # Move sample to CPU, convert to numpy, then process
            sample_cpu = sample.cpu().numpy()  # Move to CPU and convert to NumPy
            coeffs = pywt.wavedec(sample_cpu, 'db4', level=4, axis=-1)
            sample_dwt = np.concatenate(coeffs, axis=-1)
            time_freq_data.append(sample_dwt)
        time_freq_data = np.array(time_freq_data)
        time_freq_data = torch.tensor(time_freq_data).to(x.device).float()  # Convert back to tensor and send to device

        # Concatenating features from three domains along the channel dimension
        combined_data = torch.cat((time_data, freq_data, time_freq_data), dim=2)  # Check dimension
        combined_data = combined_data.transpose(1, 2)

        # CNN processing
        cnn_out = self.cnn(combined_data)
        cnn_out = F.relu(cnn_out)
        cnn_out = F.max_pool1d(cnn_out, 2)
        cnn_out = cnn_out.permute(0, 2, 1)  # Rearrange for LSTM (batch_size, seq_len, features)

        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)
        lstm_out = lstm_out[:, -1, :]  # Using the last hidden state

        # Fully connected layer
        output = self.fc(lstm_out)
        return output