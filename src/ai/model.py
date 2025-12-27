import torch
import torch.nn as nn

import src.config as config

class YahtzeeNetwork(nn.Module):
    def __init__(self, input_size=config.INPUT_SIZE, output_size=config.OUTPUT_SIZE, hidden_size=config.HIDDEN_SIZE):
        super(YahtzeeNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.network(x)
