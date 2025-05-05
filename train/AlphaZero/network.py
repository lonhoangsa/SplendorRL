# network for AlphaZero
import numpy as np
import torch
import torch.nn as nn

class AlphaZeroNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # flatten all observation components
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.policy_head = nn.Linear(256, output_dim)
        self.value_head = nn.Linear(256, 1)
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        p = self.softmax(self.policy_head(x))
        v = self.tanh(self.value_head(x)).squeeze(-1)
        return p, v 