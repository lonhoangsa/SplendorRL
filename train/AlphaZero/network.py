# network for AlphaZero
import numpy as np
import torch
import torch.nn as nn

class AlphaZeroNet(nn.Module):
    def __init__(self, input_dim, output_dim, architecture_type='simple', device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.architecture_type = architecture_type
        
        if architecture_type == 'simple':
            # Simple architecture (original)
            self.fc1 = nn.Linear(input_dim, 256)
            self.bn1 = nn.BatchNorm1d(256)
            self.fc2 = nn.Linear(256, 512)
            self.bn2 = nn.BatchNorm1d(512)
            self.fc3 = nn.Linear(512, 256)
            self.bn3 = nn.BatchNorm1d(256)
            self.policy_head = nn.Linear(256, output_dim)
            self.value_head = nn.Linear(256, 1)
        else:
            # Complex architecture with residual connections
            self.fc1 = nn.Linear(input_dim, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.fc2 = nn.Linear(512, 512)
            self.bn2 = nn.BatchNorm1d(512)
            self.fc3 = nn.Linear(512, 512)
            self.bn3 = nn.BatchNorm1d(512)
            self.fc4 = nn.Linear(512, 512)
            self.bn4 = nn.BatchNorm1d(512)
            self.fc5 = nn.Linear(512, 512)
            self.bn5 = nn.BatchNorm1d(512)
            self.policy_head = nn.Linear(512, output_dim)
            self.value_head = nn.Linear(512, 1)
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        
        # Move model to device
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on correct device
        
        if self.architecture_type == 'simple':
            x = self.relu(self.bn1(self.fc1(x)))
            x = self.relu(self.bn2(self.fc2(x)))
            x = self.relu(self.bn3(self.fc3(x)))
        else:
            # Complex architecture with residual connections
            x = self.relu(self.bn1(self.fc1(x)))
            residual = x
            x = self.relu(self.bn2(self.fc2(x)))
            x = self.relu(self.bn3(self.fc3(x)))
            x = x + residual  # Residual connection
            residual = x
            x = self.relu(self.bn4(self.fc4(x)))
            x = self.relu(self.bn5(self.fc5(x)))
            x = x + residual  # Residual connection
        
        p = self.softmax(self.policy_head(x))
        v = self.tanh(self.value_head(x)).squeeze(-1)
        return p, v

    def save(self, path):
        """Save model with architecture type"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'architecture_type': self.architecture_type
        }, path)

    @classmethod
    def load(cls, path, input_dim, output_dim, device=None):
        """Load model with architecture type"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(input_dim, output_dim, 
                   architecture_type=checkpoint['architecture_type'],
                   device=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 