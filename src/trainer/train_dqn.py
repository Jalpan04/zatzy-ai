import torch
import torch.nn as nn
import numpy as np
import random
import os
import sys
import json
from collections import deque

# Add root to path for imports
sys.path.append(os.getcwd())

from src.ai.dqn import DQNAgent
from src.game.engine import GameEngine
from src.game.scorecard import Category

def train_dqn(episodes=1000):
    agent = DQNAgent()
    
    # Replay Buffer: deque(maxlen=10000)
    # Using simple list with manual pop for now or deque
    memory = deque(maxlen=10000)
    
    optimizer = agent.optimizer
    criterion = nn.MSELoss()
    
    best_score = 0
    history = {"episode": [], "score": [], "epsilon": []}
    
    print(f"Starting DQN Training for {episodes} episodes...")
    
    for episode in range(episodes):
        engine = GameEngine()
        state = engine.get_state_vector()
        mask = engine.get_mask()
        total_score = 0
        done = False
        
        prev_score = 0
        steps = 0
        max_steps = 100 # Safety limit
        
        while not done and steps < max_steps:
            steps += 1
            # 1. Select Action
            action_type, action_val = agent.select_action(state, mask)
            
            # Map action back to index for training
            if action_type == 'keep':
                action_idx = action_val
            else:
                # Score Actions: Cat 1..13 -> Index 32..44
                # Index = Cat + 31
                action_idx = action_val + 31
                
            # 2. Step
            reward_score, valid, game_over = engine.apply_action(action_type, action_val)
            next_state = engine.get_state_vector()
            next_mask = engine.get_mask()
            
            current_game_score = engine.scorecard.get_total_score()
            done = engine.game_over
            
            # 3. Calculate Reward
            # Base Reward: Points gained this turn
            if valid:
                reward = current_game_score - prev_score
                prev_score = current_game_score
                reward = reward / 10.0 # Normalize
            else:
                # Penalty for invalid action (though mask should prevent this)
                reward = -1.0 
            
            # 4. Store Experience
            memory.append((state, action_idx, reward, next_state, done, next_mask))
            
            state = next_state
            mask = next_mask
            
            # 5. Review / Train (Mini-batch)
            if len(memory) > agent.batch_size:
                batch = random.sample(memory, agent.batch_size)
                
                # Unzip batch
                b_states, b_actions, b_rewards, b_next_states, b_dones, b_next_masks = zip(*batch)
                
                b_states = torch.FloatTensor(np.array(b_states)).to(agent.device)
                b_actions = torch.LongTensor(b_actions).unsqueeze(1).to(agent.device)
                b_rewards = torch.FloatTensor(b_rewards).unsqueeze(1).to(agent.device)
                b_next_states = torch.FloatTensor(np.array(b_next_states)).to(agent.device)
                b_dones = torch.FloatTensor(b_dones).unsqueeze(1).to(agent.device)
                b_next_masks = torch.FloatTensor(np.array(b_next_masks)).to(agent.device)
                
                # Q(s, a)
                q_values = agent.policy_net(b_states).gather(1, b_actions)
                
                # Target: r + gamma * max(Q(s', a'))
                # We must apply mask to next state Q-values so we don't accidentally pick invalid moves as max
                with torch.no_grad():
                    next_q_values = agent.target_net(b_next_states)
                    
                    # Masking next Q values
                    min_val = -1e9
                    masked_next_q = next_q_values * b_next_masks + (1 - b_next_masks) * min_val
                    
                    max_next_q = masked_next_q.max(1)[0].unsqueeze(1)
                    target_q = b_rewards + (agent.gamma * max_next_q * (1 - b_dones))
                
                loss = criterion(q_values, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        if steps >= max_steps:
             print(f"Warning: Episode {episode} hit max steps!")
             
        # End of Episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
        # Target Net Update (Soft or Hard)
        if episode % 10 == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        # Logging
        if episode % 20 == 0:
            print(f"Episode {episode}: Score {current_game_score}, Epsilon {agent.epsilon:.2f}, Memory {len(memory)}")
            
        history["episode"].append(episode)
        history["score"].append(current_game_score)
        history["epsilon"].append(agent.epsilon)
        
        if current_game_score > best_score:
            best_score = current_game_score
            # Save Checkpoint
            if not os.path.exists("checkpoints_dqn"):
                os.makedirs("checkpoints_dqn")
            torch.save(agent.policy_net.state_dict(), f"checkpoints_dqn/dqn_best_{best_score}.pth")

    # Save Log
    with open("dqn_training_log.json", "w") as f:
        json.dump(history, f)
        
if __name__ == "__main__":
    train_dqn(episodes=200) # Reduce to 200 for speed
