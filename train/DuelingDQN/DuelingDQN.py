import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, device=None):
        super(DuelingDQN, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Shared feature extraction layers with an extra layer
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),  # Added extra layer
            nn.ReLU()
        )
        # Value stream with additional layers
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),  # Added extra layer
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Advantage stream with additional layers
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),  # Added extra layer
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # Move model to device
        self.to(self.device)
        
        # Enable cuDNN benchmarking for faster training
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    def forward(self, x):
        # Ensure input is on correct device
        x = x.to(self.device)
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Combine value and advantage streams to obtain Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    