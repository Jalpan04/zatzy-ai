import os
import torch
import numpy as np
from src.game.engine import GameEngine
from src.ai.dqn import DQNAgent
from src.game.scorecard import Category

def train_dqn(episodes=1000, save_interval=100):
    env = GameEngine()
    agent = DQNAgent()
    
    # Ensure checkpoints dir exists
    os.makedirs("checkpoints_dqn", exist_ok=True)
    
    print(f"Starting DQN Training for {episodes} episodes...")
    
    scores = []
    
    for episode in range(1, episodes + 1):
        env = GameEngine() # Reset environment
        state = env.get_state_vector()
        mask = env.get_mask()
        total_score = 0
        done = False
        
        while not done:
            # Select Action
            action_idx = agent.select_action(state, mask)
            
            # Decode Action
            # 0-31: Keep Masks
            # 32-44: Categories
            if action_idx < 32:
                action_type = "keep"
                action_val = action_idx
            else:
                action_type = "score"
                action_val = action_idx - 32
                
            # Step
            # We need to capture reward. 
            # In Yahtzee, reward = points gained this turn.
            prev_score = env.scorecard.get_total_score()
            env.apply_action(action_type, action_val)
            new_score = env.scorecard.get_total_score()
            
            reward = new_score - prev_score
            
            # Custom Rewards (Shaping)
            # Encouraging Yahtzee?
            if action_type == "score" and action_val == Category.YAHTZEE and reward == 50:
                reward += 50 # Bonus
            
            # Normalize Reward for stability? (Optional, /10 or /50)
            reward /= 10.0 
            
            next_state = env.get_state_vector()
            next_mask = env.get_mask()
            done = env.game_over
            
            # Store transition
            agent.memory.push(state, action_idx, reward, next_state, done, mask, next_mask)
            
            # Learn
            agent.update()
            
            state = next_state
            mask = next_mask
            
        scores.append(env.scorecard.get_total_score())
        agent.decay_epsilon()
        
        if episode % 10 == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            avg_score = np.mean(scores[-10:])
            print(f"Episode {episode} | Avg Score: {avg_score:.1f} | Epsilon: {agent.epsilon:.2f}")
            
        if episode % save_interval == 0:
             path = f"checkpoints_dqn/dqn_ep_{episode}_score_{int(avg_score)}.pkl"
             torch.save(agent.policy_net.state_dict(), path)
             print(f"Saved checkpoint: {path}")

if __name__ == "__main__":
    train_dqn()
