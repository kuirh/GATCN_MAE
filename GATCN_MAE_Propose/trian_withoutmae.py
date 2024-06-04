# 创建Dataset实例
import os

import pandas as pd
import wandb

import numpy as np
from torch.utils.data import DataLoader

from Build_Dataset import PoseDataset


from torch import nn
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from Build_Modal import ATGCNModel, MaskedAutoencoder,LSTMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score,matthews_corrcoef



sweep_configuration = {
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        #data parameters
        "atgcn_learning_rate": {"values": [0.00001]},
        "cls_learning_rate": {"values": [0.00005]},
        "gat":{"values": ['gat']},
        "st_attention":{"values": [True]},
}
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="veeg")

def train(config=None):
    with wandb.init(name='pose_withoumae'):


        upper_body_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]
        edges = [
            (0, 1), (0, 4),
            (1, 2), (2, 3),
            (4, 5), (5, 6),
            (7, 1), (8, 4),
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15),
            (12, 14), (14, 16)
        ]
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        # 转换为 `edge_index`
        edge_index = []
        for (i, j) in edges:
            if i in upper_body_landmarks and j in upper_body_landmarks:
                edge_index.append([upper_body_landmarks.index(i), upper_body_landmarks.index(j)])
                edge_index.append([upper_body_landmarks.index(j), upper_body_landmarks.index(i)])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_index= edge_index.to(device)
        model_save_path= 'models'
        os.makedirs(model_save_path, exist_ok=True)


        #数据集
        dataset = PoseDataset(folder_path='../output_folder', upper_body_landmarks=upper_body_landmarks)



        # 首先提取标签
        labels = [dataset[i][1].item() for i in range(len(dataset))]  # 使用 .item() 获取 Tensor 中的值

        # 使用train_test_split进行分层抽样拆分
        train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, stratify=labels)

        # 创建对应的Subset
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        # 创建DataLoader实例
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        # 模型参数设置
        tempcov1_model_parameters = {
            'hidden_channel': 64,
            'kernel_size': (9,1),
            'padding': (4,0),
            'hidden_num_layers': 1,
            'activation': 'leaky_relu'
        }

        GAT_model_parameters = {
            'out_channels': 64,
            'hidden_channels': 64,
            'hidden_num_layers': 1,
            'heads': 2,
            'activation': 'leaky_relu'
        }

        tempcov2_model_parameters = {
            'hidden_channel': 64,
            'kernel_size': (15,1),
            'padding': (7,0),
            'hidden_num_layers': 1,
            'activation': 'leaky_relu'
        }

        LSTM_model_parameters = {
            'hidden_features_num': 128,
            'out_features_num': 256,
            'hidden_num_layers': 1,
            'dropout': 0.1,
            'activation': 'leaky_relu'
        }

        ProposeModel_parameters = {
            'use_bactchnorm': False,
            'use_gating': False,

            'use_cbam': wandb.config.st_attention,
            'gcnorgat': wandb.config.gat,

        }

        atgcn_moael=ATGCNModel(in_channel=4,
                                  tempcov1_model_parameters=tempcov1_model_parameters,
                                  GAT_model_parameters=GAT_model_parameters,
                                  tempcov2_model_parameters=tempcov2_model_parameters,
                                  LSTM_model_parameters=LSTM_model_parameters,
                                  ProposeModel_parameters=ProposeModel_parameters).to(device)

        cls_head=LSTMClassifier(tempcov2_model_parameters['hidden_channel']* 19,LSTM_model_parameters).to(device)


        #optimizer
        classification_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]).to(device))
        #optimizer = optim.Adam(list(mae_model.parameters())+list(atgcn_moael.parameters())+list(cls_head.parameters()), lr=0.00005)
        optimizer = optim.Adam([
            {'params': atgcn_moael.parameters(), 'lr': wandb.config.atgcn_learning_rate},
            {'params': cls_head.parameters(), 'lr': wandb.config.cls_learning_rate}
        ])

        #评估模型
        def evaluate(atgcn_model, cls_head, dataloader, device):
            atgcn_model.eval()
            cls_head.eval()

            all_preds = []
            all_labels = []
            #starttime = time.time()

            with torch.no_grad():
                for data, labels in dataloader:
                    data, labels = data.to(device), labels.to(device).float().unsqueeze(1)

                    atgcn_feature = atgcn_model(data, edge_index)

                    outputs = cls_head(atgcn_feature)
                    preds = torch.round(torch.sigmoid(outputs))
                    #wandb.log({"compute_time": time.time() - starttime})



                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            mcc= matthews_corrcoef(all_labels, all_preds)
            auc_roc = roc_auc_score(all_labels, all_preds)

            return accuracy, precision, recall, f1, mcc, auc_roc,all_preds,all_labels



        # 训练模型
        num_epochs = 100
        best_val_accuracy = 0.0  # 用于跟踪最佳验证准确率

        for epoch in range(num_epochs):
            atgcn_moael.train()
            cls_head.train()

            running_loss = 0.0
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()

                atgcn_feature=atgcn_moael(data, edge_index)
                outputs = cls_head(atgcn_feature)

                classification_loss = classification_criterion(outputs, labels)
                wandb.log({"classification_loss": classification_loss.item()})
                loss = classification_loss
                wandb.log({"bacth_loss": loss.item()})
                loss.backward()
                optimizer.step()





                running_loss += loss.item() * data.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            wandb.log({"epoch_loss": epoch_loss, "epoch": epoch})


        print('Training complete.')


        # 在训练完后评估模型
        val_accuracy, precision, recall, f1, mcc, auc_roc,preds,labels = evaluate(atgcn_moael, cls_head, val_loader, device)
        wandb.log({
                "val_accuracy": val_accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mcc": mcc,
                "auc_roc": auc_roc
            })

wandb.agent(sweep_id, train, count=5)





