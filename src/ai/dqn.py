import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from src.ai.model import YahtzeeNetwork
import src.config as config

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.append((state, action, reward, next_state, done, mask, next_mask))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, mask, next_mask = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done), np.array(mask), np.array(next_mask))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, learning_rate=1e-3, gamma=0.99, buffer_size=100000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Networks (Policy and Target)
        self.policy_net = YahtzeeNetwork(config.INPUT_SIZE, config.OUTPUT_SIZE, config.HIDDEN_SIZE).to(self.device)
        self.target_net = YahtzeeNetwork(config.INPUT_SIZE, config.OUTPUT_SIZE, config.HIDDEN_SIZE).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        
        self.gamma = gamma
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state, mask, training=True):
        # Epsilon-Greedy logic
        if training and random.random() < self.epsilon:
            # Random legal move
            legal_indices = np.where(mask == 1)[0]
            if len(legal_indices) == 0: return 0 # Should not happen
            return int(random.choice(legal_indices))
        
        # Greedy logic
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
            
            q_values = self.policy_net(state_t)
            
            # Apply Mask (Set illegal moves to -infinity)
            q_values = q_values.masked_fill(mask_t == 0, -float('inf'))
            
            return q_values.argmax().item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones, masks, next_masks = self.memory.sample(self.batch_size)
        
        # Convert to Tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        next_masks = torch.FloatTensor(next_masks).to(self.device)
        
        # Current Q values
        curr_q = self.policy_net(states).gather(1, actions)
        
        # Next Q values (Target)
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            # Mask illegal moves in next state
            next_q_values = next_q_values.masked_fill(next_masks == 0, -float('inf'))
            next_max_q = next_q_values.max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_max_q
            
        # Loss
        loss = nn.MSELoss()(curr_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
