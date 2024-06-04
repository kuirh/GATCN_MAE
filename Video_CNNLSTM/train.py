import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

from Build_Dataset import VideoDataset
from Build_Network import TCN_LSTM


# 初始化数据集和DataLoader
def train():
    warnings.filterwarnings('ignore', category=UserWarning)
    dataset = VideoDataset(folder_path='video_segments')
    labels = [dataset[i][1].item() for i in range(len(dataset))]  # 提取标签
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, stratify=labels)

    #train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2)

    # 分割数据集

    # 生成训练和验证子集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型实例化并转移到设备
    model = TCN_LSTM().to(device)
    #model=  EEGHybridNet().to(device)

    # 定义损失函数和优化器

    pos_weight = torch.tensor([13]).to(device)  # 根据你的类不平衡情况调整权重
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    def train(model, device, train_loader, criterion, optimizer):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device).float().unsqueeze(1)  # 确保标签为浮点数
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def validate(model, device, val_loader, criterion):
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        #starttime = time.time()
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device).float().unsqueeze(1)  # 确保标签为浮点数
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.round(torch.sigmoid(outputs))
                #print({"compute_time": time.time() - starttime})

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算平均损失
        avg_loss = total_loss / len(val_loader)

        # 将收集的预测和标签转换为 NumPy 数组
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 计算各项评价指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        auc_roc = roc_auc_score(all_labels, all_preds)  # 注意：AUC 计算可能需要预测得分而非二值化结果

        return avg_loss, accuracy, precision, recall, f1, mcc, auc_roc


    # 训练和验证模型
    num_epochs = 200
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer)
        print(
            f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}%')

    val_loss, val_accuracy, val_precision, val_recall, val_f1, val_mcc, val_auc_roc = validate(model, device,
                                                                                               val_loader, criterion)
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy*100:.4f}%, Val Precision: {val_precision:.2f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}, Val MCC: {val_mcc:.4f}, Val AUC-ROC: {val_auc_roc:.4f}')


for i in range(10):
    train()