from torch import nn
import torch
from torch_geometric.data import Data, Batch
import torch_geometric.nn as  torch_geometric

def choose_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'none':
        return None
    else:
        raise ValueError('Activation function not supported')

class GatingLayer(nn.Module):
    def __init__(self, in_features, reduction_features=4):
        super(GatingLayer, self).__init__()
        self.in_features = in_features
        self.reduction_features = reduction_features
        self.fc = nn.Sequential(
            nn.Linear(in_features * 2, reduction_features),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_features, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        mean = x.mean(dim=[2, 3])
        std = x.std(dim=[2, 3])
        stats = torch.cat((mean, std), dim=1)
        gate = self.fc(stats)
        return x * gate.unsqueeze(2).unsqueeze(3), gate


class GATModule(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels,hidden_num_layers=1,heads=2,activation='none'):
        super(GATModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.hidden_num_layers = hidden_num_layers
        self.convs = nn.ModuleList()
        self.convs.append(torch_geometric.GATConv(in_channels, hidden_channels,heads=heads,concat=False))
        for _ in range(hidden_num_layers):
            self.convs.append(torch_geometric.GATConv(hidden_channels, hidden_channels,heads=heads,concat=False))
        self.convs.append(torch_geometric.GATConv(hidden_channels, out_channels,heads=heads,concat=False))
        self.activation = choose_activation(activation)




    def forward(self, x, edge_index):
        # Apply first GCN layer and ReLU activation
        for i, conv in enumerate(self.convs):
            if i==0:
                if self.in_channels==self.hidden_channels:
                    x = conv(x, edge_index)+x
                else:
                    x = conv(x, edge_index)
            elif i==(len(self.convs)-1):
                if self.hidden_channels==self.out_channels:
                    x = conv(x, edge_index)+x
                    if self.activation is not None:
                        x = self.activation(x)
                else:
                    x = conv(x, edge_index)+x
            else:
                x = conv(x, edge_index)+x



        return x

class GCNModule(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, hidden_num_layers=1, activation='none'):
        super(GCNModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.hidden_num_layers = hidden_num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(hidden_num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.activation = choose_activation(activation)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            if i == 0:
                x = conv(x, edge_index)
            elif i == (len(self.convs) - 1):
                x = conv(x, edge_index)
                if self.activation is not None:
                    x = self.activation(x)
            else:
                x = conv(x, edge_index)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_features_num, hidden_features_num=40, out_features_num=80, hidden_num_layers=1, dropout=0.1, activation='none'):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_features_num, hidden_features_num, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_features_num, hidden_features_num, num_layers=hidden_num_layers, batch_first=True, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_features_num, out_features_num, batch_first=True)

    def forward(self, x):
        x, (hn, cn) = self.lstm1(x)
        x = self.dropout1(x)
        x, (hn, cn) = self.lstm(x)
        x = self.dropout2(x)
        x, (hn, cn) = self.lstm2(x)
        x = x[:, -1, :]
        return x

class ClassificationHead(nn.Module):
    def __init__(self, input_dim):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

class ATGCNModel(nn.Module):
    def __init__(self, in_channel, tempcov1_model_parameters, GAT_model_parameters, tempcov2_model_parameters, LSTM_model_parameters, ProposeModel_parameters):
        super(ATGCNModel, self).__init__()
        self.gcnorgat = ProposeModel_parameters['gcnorgat']
        self.use_bacth_norm = ProposeModel_parameters['use_bactchnorm']
        self.use_gating = ProposeModel_parameters['use_gating']
        self.batch_norm = nn.BatchNorm2d(in_channel)
        self.gating = GatingLayer(in_channel)
        self.temp_conv1_in = nn.Conv2d(in_channel, tempcov1_model_parameters['hidden_channel']-in_channel, kernel_size=tempcov1_model_parameters['kernel_size'], padding=tempcov1_model_parameters['padding'])
        self.temp_conv1_hidden = nn.ModuleList()
        for _ in range(tempcov1_model_parameters['hidden_num_layers']):
            self.temp_conv1_hidden.append(nn.Conv2d(tempcov1_model_parameters['hidden_channel'], tempcov1_model_parameters['hidden_channel'], kernel_size=tempcov1_model_parameters['kernel_size'], padding=tempcov1_model_parameters['padding']))
        self.temp_conv1_activation = choose_activation(tempcov1_model_parameters['activation'])
        self.gcn = GCNModule(in_channels=tempcov1_model_parameters['hidden_channel'],
                             out_channels=GAT_model_parameters['out_channels'],
                             hidden_channels=GAT_model_parameters['hidden_channels'],
                             hidden_num_layers=GAT_model_parameters['hidden_num_layers'],
                             activation=GAT_model_parameters['activation']) if self.gcnorgat=='gcn' else GATModule(in_channels=tempcov1_model_parameters['hidden_channel'],
                                out_channels=GAT_model_parameters['out_channels'],
                                hidden_channels=GAT_model_parameters['hidden_channels'],
                                hidden_num_layers=GAT_model_parameters['hidden_num_layers'],
                                heads=GAT_model_parameters['heads'],
                                activation=GAT_model_parameters['activation'])



        self.temp_conv2_in = nn.Conv2d(GAT_model_parameters['out_channels'], tempcov2_model_parameters['hidden_channel'], kernel_size=tempcov2_model_parameters['kernel_size'], padding=tempcov2_model_parameters['padding'])
        self.temp_conv2_hidden = nn.ModuleList()
        for _ in range(tempcov2_model_parameters['hidden_num_layers']):
            self.temp_conv2_hidden.append(nn.Conv2d(tempcov2_model_parameters['hidden_channel'], tempcov2_model_parameters['hidden_channel'], kernel_size=tempcov2_model_parameters['kernel_size'], padding=tempcov2_model_parameters['padding']))
        self.temp_conv2_activation = choose_activation(tempcov2_model_parameters['activation'])

    def forward(self, x, edge_index):
        x = x.permute(0, 3, 1, 2)
        if self.use_bacth_norm:
            x = self.batch_norm(x)
        if self.use_gating:
            x, _ = self.gating(x)
        x = torch.cat((x, self.temp_conv1_in(x)), dim=1)
        for conv in self.temp_conv1_hidden:
            x = conv(x) + x
        if self.temp_conv1_activation is not None:
            x = self.temp_conv1_activation(x)
        x = x.permute(0, 2, 3, 1)
        batch_size, time_steps, num_nodes, node_features = x.size()
        x_flattened = x.view(batch_size * time_steps, num_nodes, -1)
        data_list = [Data(x=x_flattened[i], edge_index=edge_index) for i in range(x_flattened.size(0))]
        batched_data = Batch.from_data_list(data_list)
        gcn_outputs = self.gcn(batched_data.x, batched_data.edge_index)
        gcn_outputs = gcn_outputs.view(batch_size, time_steps, num_nodes, -1)
        x = gcn_outputs + x
        x = x.permute(0, 3, 1, 2)
        temp_conv2_outputs = [x]
        temp_conv2 = self.temp_conv2_in(x)
        temp_conv2_outputs.append(temp_conv2)
        for conv in self.temp_conv2_hidden:
            temp_conv2 = conv(temp_conv2)
            temp_conv2_outputs.append(temp_conv2)
        x = sum(temp_conv2_outputs)
        if self.temp_conv2_activation is not None:
            x = self.temp_conv2_activation(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(batch_size, time_steps, -1)

        return x

class LSTMClassifier(nn.Module):
    def __init__(self, inchannel,LSTM_model_parameters):
        super(LSTMClassifier, self).__init__()
        self.lstm = LSTMModel(input_features_num=inchannel, hidden_features_num=LSTM_model_parameters['hidden_features_num'], out_features_num=LSTM_model_parameters['out_features_num'], hidden_num_layers=LSTM_model_parameters['hidden_num_layers'], dropout=LSTM_model_parameters['dropout'], activation=LSTM_model_parameters['activation'])
        self.classifier = ClassificationHead(LSTM_model_parameters['out_features_num'])

    def forward(self, x):
        x = self.lstm(x)
        x = self.classifier(x)
        return x


class MaskedAutoencoder(nn.Module):
    def __init__(self, combined_features, hidden_feature_size, num_encoder_layers, num_decoder_layers, mask_ratio=0.75):
        super(MaskedAutoencoder, self).__init__()
        self.mask_ratio = mask_ratio
        self.combined_features = combined_features
        self.hidden_feature_size = hidden_feature_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        self.encoder = nn.Sequential(*[
            nn.Linear(combined_features, hidden_feature_size) if i == 0 else nn.Linear(hidden_feature_size,
                                                                                       hidden_feature_size)
            for i in range(num_encoder_layers)
        ])
        self.decoder = nn.Sequential(*[
                                          nn.Linear(hidden_feature_size, hidden_feature_size)
                                          for i in range(num_decoder_layers - 1)
                                      ] + [nn.Linear(hidden_feature_size, combined_features)])

    def forward(self, x):
        device = x.device
        batch_size, timestep, num_node, num_feature = x.shape

        # Ensure mask is the same shape as x
        mask = torch.rand(batch_size, timestep, num_node, num_feature, device=device) < self.mask_ratio
        masked_x = x.clone()
        masked_x[mask] = 0

        x_flattened = x.view(batch_size, timestep, num_node * num_feature)
        masked_x_flattened = masked_x.view(batch_size, timestep, num_node * num_feature)

        # Encode and decode
        encoded = masked_x_flattened
        for layer in self.encoder:
            encoded = layer(encoded)
        decoded = encoded
        for layer in self.decoder:
            decoded = layer(decoded)

        decoded = decoded.view(batch_size, timestep, num_node, num_feature)

        return decoded, encoded, mask