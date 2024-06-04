import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

upper_body_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]

class PoseDataset(Dataset):
    def __init__(self, folder_path, upper_body_landmarks, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.upper_body_landmarks = upper_body_landmarks

        self.data_files = []
        self.labels = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    label = 0 if '_norm' in file_path else 1
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
        df = pd.read_csv(csv_path)

        if 'landmark_id' not in df.columns:
            raise KeyError(f"'landmark_id' not found in columns of {csv_path}")

        df = df[df['landmark_id'].isin(self.upper_body_landmarks)]

        timesteps = df['frame'].nunique()
        node_num = len(self.upper_body_landmarks)
        node_feature = 4

        data = np.zeros((timesteps, node_num, node_feature))

        for i, frame in enumerate(sorted(df['frame'].unique())):
            frame_data = df[df['frame'] == frame]
            for j, landmark_id in enumerate(self.upper_body_landmarks):
                landmark_data = frame_data[frame_data['landmark_id'] == landmark_id]
                if not landmark_data.empty:
                    data[i, j, 0] = landmark_data['x'].values[0]
                    data[i, j, 1] = landmark_data['y'].values[0]
                    data[i, j, 2] = landmark_data['z'].values[0]
                    data[i, j, 3] = landmark_data['visibility'].values[0]

        return data


# 上半身关键点索引
