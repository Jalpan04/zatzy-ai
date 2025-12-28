import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.output(x)

class DQNAgent:
    def __init__(self, input_size=48, output_size=45, device='cpu'):
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        
        self.policy_net = DQN(input_size, output_size).to(device)
        self.target_net = DQN(input_size, output_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = [] # Replay Buffer
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def select_action(self, state, mask, engine=None):
        # state is np array
        if random.random() < self.epsilon:
            # Random valid action
            # Mask indices where mask == 1 (legal)
            valid_indices = np.where(mask == 1)[0]
            if len(valid_indices) == 0: return ("score", 0) # Fallback
            action_idx = random.choice(valid_indices)
        else:
            # Greedy action
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                
                # Apply mask: Set invalid actions to -inf
                mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
                min_val = torch.min(q_values) - 10000.0 # Safe large negative
                masked_q = q_values * mask_t + (1 - mask_t) * min_val
                
                action_idx = masked_q.argmax().item()
        
        # Decode Action
        # 0-31: Keep (Mask)
        # 32-44: Score (Category 0-12)
        if action_idx < 32:
            return ("keep", action_idx)
        else:
            # Indices 32-44 map to Categories 1-13
            # Action 32 -> Cat 1
            # Action 44 -> Cat 13
            return ("score", action_idx - 32 + 1)
