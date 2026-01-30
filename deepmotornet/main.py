import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepMotorNet(nn.Module):
    def __init__(self, n_channels=129, seq_length=1000, config=None):
        super(DeepMotorNet, self).__init__()
        
        default_config = {
            'conv1_filters': 32,
            'conv2_filters': 64,
            'conv3_filters': 128,
            'conv4_filters': 256,
            'kernel1': 31,
            'kernel2': 15,
            'kernel3': 7,
            'kernel4': 3,
            'dropout1': 0.3,
            'dropout2': 0.3,
            'dropout3': 0.3,
            'dropout_fc': 0.5,
            'fc1_size': 256,
            'fc2_size': 128,
            'use_batch_norm': True,
            'use_residual': False
        }
        
        if config:
            default_config.update(config)
        self.config = default_config
        
        self.conv1 = nn.Conv1d(n_channels, self.config['conv1_filters'], 
                              kernel_size=self.config['kernel1'], 
                              padding=self.config['kernel1']//2)
        if self.config['use_batch_norm']:
            self.bn1 = nn.BatchNorm1d(self.config['conv1_filters'])
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(self.config['conv1_filters'], self.config['conv2_filters'],
                              kernel_size=self.config['kernel2'],
                              padding=self.config['kernel2']//2)
        if self.config['use_batch_norm']:
            self.bn2 = nn.BatchNorm1d(self.config['conv2_filters'])
        self.pool2 = nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(self.config['conv2_filters'], self.config['conv3_filters'],
                              kernel_size=self.config['kernel3'],
                              padding=self.config['kernel3']//2)
        if self.config['use_batch_norm']:
            self.bn3 = nn.BatchNorm1d(self.config['conv3_filters'])
        
        if self.config['conv4_filters'] > 0:
            self.conv4 = nn.Conv1d(self.config['conv3_filters'], self.config['conv4_filters'],
                                  kernel_size=self.config['kernel4'],
                                  padding=self.config['kernel4']//2)
            if self.config['use_batch_norm']:
                self.bn4 = nn.BatchNorm1d(self.config['conv4_filters'])
            final_channels = self.config['conv4_filters']
        else:
            final_channels = self.config['conv3_filters']
        
        self.pool3 = nn.AdaptiveAvgPool1d(10)
        
        with torch.no_grad():
            dummy = torch.randn(1, n_channels, seq_length)
            dummy = self.pool1(F.relu(self.bn1(self.conv1(dummy)) if self.config['use_batch_norm'] else self.conv1(dummy)))
            dummy = self.pool2(F.relu(self.bn2(self.conv2(dummy)) if self.config['use_batch_norm'] else self.conv2(dummy)))
            dummy = F.relu(self.bn3(self.conv3(dummy)) if self.config['use_batch_norm'] else self.conv3(dummy))
            if self.config['conv4_filters'] > 0:
                dummy = F.relu(self.bn4(self.conv4(dummy)) if self.config['use_batch_norm'] else self.conv4(dummy))
            dummy = self.pool3(dummy)
            self.flattened_size = dummy.view(1, -1).shape[1]
        
        self.fc1 = nn.Linear(self.flattened_size, self.config['fc1_size'])
        if self.config['use_batch_norm']:
            self.fc_bn1 = nn.BatchNorm1d(self.config['fc1_size'])
        
        self.fc2 = nn.Linear(self.config['fc1_size'], self.config['fc2_size'])
        if self.config['use_batch_norm']:
            self.fc_bn2 = nn.BatchNorm1d(self.config['fc2_size'])
        
        self.fc3 = nn.Linear(self.config['fc2_size'], 2)
        
        self.dropout_conv1 = nn.Dropout(self.config['dropout1'])
        self.dropout_conv2 = nn.Dropout(self.config['dropout2'])
        self.dropout_conv3 = nn.Dropout(self.config['dropout3'])
        self.dropout_fc1 = nn.Dropout(self.config['dropout_fc'])
        self.dropout_fc2 = nn.Dropout(self.config['dropout_fc'])
        
        print(f"DeepMotorNet configuration: {self.config}")
        print(f"Flattened size: {self.flattened_size}")
    
    def forward(self, x):
        self.activations = {}
        
        x1 = self.conv1(x)
        if self.config['use_batch_norm']:
            x1 = self.bn1(x1)
        self.activations['conv1'] = x1
        x1 = self.dropout_conv1(self.pool1(F.relu(x1)))
        
        x2 = self.conv2(x1)
        if self.config['use_batch_norm']:
            x2 = self.bn2(x2)
        self.activations['conv2'] = x2
        x2 = self.dropout_conv2(self.pool2(F.relu(x2)))
        
        x3 = self.conv3(x2)
        if self.config['use_batch_norm']:
            x3 = self.bn3(x3)
        self.activations['conv3'] = x3
        x3 = self.dropout_conv3(F.relu(x3))
        
        if self.config['conv4_filters'] > 0:
            x4 = self.conv4(x3)
            if self.config['use_batch_norm']:
                x4 = self.bn4(x4)
            self.activations['conv4'] = x4
            x3 = F.relu(x4)
        
        x = self.pool3(x3)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        if self.config['use_batch_norm']:
            x = self.fc_bn1(x)
        x = self.dropout_fc1(F.relu(x))
        
        x = self.fc2(x)
        if self.config['use_batch_norm']:
            x = self.fc_bn2(x)
        x = self.dropout_fc2(F.relu(x))
        
        x = self.fc3(x)
        return x
    
    def get_gradients(self, input_data, target_class=None):
        self.eval()
        input_data = input_data.clone().requires_grad_(True)
        
        output = self(input_data)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        self.zero_grad()
        
        one_hot_output = torch.zeros_like(output)
        one_hot_output.scatter_(1, target_class.unsqueeze(1), 1.0)
        
        output.backward(gradient=one_hot_output)
        
        gradients = input_data.grad
        
        return gradients
