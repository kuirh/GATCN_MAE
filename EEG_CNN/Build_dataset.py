import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform

        # 收集所有 csv 文件的路径和相应的标签
        self.data_files = []
        self.labels = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    label = 0 if '_normal' in root else 1  # 根据文件路径中是否包含 '_normal' 判断标签
                    self.data_files.append(file_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        label = self.labels[idx]
        data = self.read_csv_to_ndarray(file_path)

        if self.transform:
            data = self.transform(data)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def read_csv_to_ndarray(self, csv_path):
        # 读取 CSV 文件到 DataFrame
        df = pd.read_csv(csv_path)

        # 选择你需要的列或进行其他数据处理
        # 假设每个文件包含一个矩阵，我们将整个文件读取为 numpy 数组
        df = df.iloc[:, 1:]
        data = df.values

        return data

# 上半身关键点索引
