import os

import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class VideoDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.data_files = []
        self.labels = []

        # Scan for video files and label based on filename
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.mp4'):
                    file_path = os.path.join(root, file)
                    label = 0 if '_norm' in file_path else 1
                    self.data_files.append(file_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        label = self.labels[idx]
        data = self.read_mp4_to_ndarray(file_path)

        if self.transform:
            data = self.transform(data)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def read_mp4_to_ndarray(self, file_path):
        cap = cv2.VideoCapture(file_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Optionally, resize or apply other preprocessing here
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (144,96))
            frames.append(frame)

        cap.release()
        data = np.array(frames)
        return data


# 上半身关键点索引
