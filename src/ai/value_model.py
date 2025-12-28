
import torch
import torch.nn as nn
import src.config as config

class ValueNetwork(nn.Module):
    def __init__(self, input_size=config.INPUT_SIZE, hidden_size=1024):
        super(ValueNetwork, self).__init__()
        # Deep Architecture: 1024 -> 512 -> 256 -> 1
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2), # Regularization
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            
            nn.Linear(hidden_size // 4, 1) # Regression Output
        )
        
    def forward(self, x):
        return self.network(x)
