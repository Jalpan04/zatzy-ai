import torch
import torch.nn as nn
import numpy as np
import random
import os
import sys
import json
from collections import deque
import time

# Add root to path for imports
sys.path.append(os.getcwd())

from src.ai.dqn import DQNAgent
from src.game.engine import GameEngine
from src.game.scorecard import Category

def calculate_shaped_reward(engine, action_type, action_val, valid, prev_score):
    """
    Advanced Reward Function to guide the agent towards strategy.
    """
    if not valid:
        return -10.0 # Heavy penalty for invalid moves (though mask handles this)

    current_score = engine.scorecard.get_total_score()
    raw_gain = current_score - prev_score
    
    reward = raw_gain / 10.0 # Base normalization
    
    # --- STRATEGY BONUSES ---
    if action_type == 'score':
        # 1. Yahtzee Bonus
        if action_val == Category.YAHTZEE and raw_gain >= 50:
             reward += 10.0 # HUGE bonus for getting a Yahtzee
        
        # 2. Upper Section Bonus awareness
        # If filling an upper section with max points (e.g. 5 Fives = 25)
        if action_val <= 6:
             max_possible = action_val * 5
             if raw_gain >= max_possible:
                 reward += 2.0
        
        # 3. Penalty for Zeroing valuable categories
        # If we scored 0 in Large Straight or Yahtzee
        if raw_gain == 0:
            if action_val in [Category.LARGE_STRAIGHT, Category.YAHTZEE]:
                reward -= 5.0 # Terrible move
            elif action_val in [Category.SMALL_STRAIGHT, Category.FULL_HOUSE]:
                reward -= 2.0
    
    return reward

def train_dqn_pro(episodes=5000):
    agent = DQNAgent()
    
    # Advanced Hyperparameters
    agent.epsilon = 1.0
    agent.epsilon_decay = 0.9995 # Slower decay for more exploration
    agent.epsilon_min = 0.05
    agent.batch_size = 64
    
    # Replay Buffer: Larger
    memory = deque(maxlen=50000)
    
    optimizer = agent.optimizer
    criterion = nn.MSELoss()
    
    best_score = 0
    history = {"episode": [], "score": [], "epsilon": [], "avg_loss": []}
    
    print(f"Starting DQN PRO Training for {episodes} episodes...")
    start_time = time.time()
    
    for episode in range(episodes):
        engine = GameEngine()
        state = engine.get_state_vector()
        mask = engine.get_mask()
        
        total_score = 0
        prev_score = 0
        
        steps = 0
        max_steps = 100 
        episode_loss = []
        
        done = False
        
        while not done and steps < max_steps:
            steps += 1
            
            # Select Action
            action_type, action_val = agent.select_action(state, mask)
            
            # Action Mapping
            if action_type == 'keep':
                action_idx = action_val
            else:
                action_idx = action_val + 31
                
            # Step
            reward_score_val, valid, game_over = engine.apply_action(action_type, action_val)
            next_state = engine.get_state_vector()
            next_mask = engine.get_mask()
            done = engine.game_over
            
            # calculate Shaped Reward
            reward = calculate_shaped_reward(engine, action_type, action_val, valid, prev_score)
            prev_score = engine.scorecard.get_total_score()
            
            # Store
            memory.append((state, action_idx, reward, next_state, done, next_mask))
            
            state = next_state
            mask = next_mask
            
            # Train
            if len(memory) > agent.batch_size:
                batch = random.sample(memory, agent.batch_size)
                b_states, b_actions, b_rewards, b_next_states, b_dones, b_next_masks = zip(*batch)
                
                b_states = torch.FloatTensor(np.array(b_states)).to(agent.device)
                b_actions = torch.LongTensor(b_actions).unsqueeze(1).to(agent.device)
                b_rewards = torch.FloatTensor(b_rewards).unsqueeze(1).to(agent.device)
                b_next_states = torch.FloatTensor(np.array(b_next_states)).to(agent.device)
                b_dones = torch.FloatTensor(b_dones).unsqueeze(1).to(agent.device)
                b_next_masks = torch.FloatTensor(np.array(b_next_masks)).to(agent.device)
                
                # Double DQN Logic (Optional, stick to regular Target Net for now for simplicity/speed)
                # Q(s, a)
                q_values = agent.policy_net(b_states).gather(1, b_actions)
                
                with torch.no_grad():
                    next_q_values = agent.target_net(b_next_states)
                    min_val = -1e9
                    masked_next_q = next_q_values * b_next_masks + (1 - b_next_masks) * min_val
                    max_next_q = masked_next_q.max(1)[0].unsqueeze(1)
                    target_q = b_rewards + (agent.gamma * max_next_q * (1 - b_dones))
                
                loss = criterion(q_values, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                episode_loss.append(loss.item())

        # End Episode handling
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
        if episode % 10 == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        final_score = engine.scorecard.get_total_score()
        
        # Logging
        history["episode"].append(episode)
        history["score"].append(final_score)
        history["epsilon"].append(agent.epsilon)
        avg_loss = sum(episode_loss)/len(episode_loss) if episode_loss else 0
        history["avg_loss"].append(avg_loss)
        
        if episode % 100 == 0:
            print(f"Ep {episode}: Score {final_score}, Eps {agent.epsilon:.3f}, AvgLoss {avg_loss:.4f}, Best {best_score}")
            # Live Dashboard Update
            with open("dqn_training_log.json", "w") as f:
                json.dump(history, f)
        
        if final_score > best_score:
            best_score = final_score
            if not os.path.exists("checkpoints_dqn"):
                os.makedirs("checkpoints_dqn")
            torch.save(agent.policy_net.state_dict(), f"checkpoints_dqn/dqn_pro_best_{best_score}.pth")
            print(f"*** NEW BEST: {best_score} ***")

    print(f"Training Complete. Time: {time.time() - start_time:.2f}s")
    
    # Save Log
    with open("dqn_training_log.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes")
    args = parser.parse_args()
    
    train_dqn_pro(episodes=args.episodes)
