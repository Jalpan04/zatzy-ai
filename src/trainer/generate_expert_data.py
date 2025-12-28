import numpy as np
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.game.engine import GameEngine
from src.ai.expectimax import ExpectimaxAgent
from src.game.scorecard import Category

def generate_data(num_games=500, output_file="expert_data.npz"):
    print(f"Generating data using Expectimax for {num_games} games...")
    
    agent = ExpectimaxAgent()
    
    all_states = []
    all_actions = []
    
    # Mapping
    # Score 1..13 -> 0..12
    # Keep 0..31 -> 13..44
    
    for game_idx in range(num_games):
        engine = GameEngine()
        
        while not engine.game_over:
            # 1. Get State
            state = engine.get_state_vector()
            
            # 2. Get Expert Action
            # Expectimax doesn't use the mask arg, but we pass it anyway.
            # Using get_mask() instead of get_valid_actions_mask()
            mask = engine.get_mask()
            action_type, action_val = agent.select_action(state, mask, engine)
            
            # 3. Encode Action
            # Layout must match GameEngine.get_mask(): [Keep(32), Score(13)]
            target_idx = -1
            if action_type == 'keep':
                 # Keep mask is 0..31. Index is just the value.
                 target_idx = action_val 
            else:
                 # Score category is 1..13. Index starts at 32.
                 # Category.ONES(1) -> 32
                 target_idx = 32 + (action_val - 1)
                
            all_states.append(state)
            all_actions.append(target_idx)
            
            # 4. Apply Action
            engine.apply_action(action_type, action_val)
            
        if (game_idx + 1) % 50 == 0:
            print(f"Completed {game_idx + 1}/{num_games} games. Total samples: {len(all_states)}")
            # Periodic Save
            np.savez_compressed(output_file, states=np.array(all_states, dtype=np.float32), actions=np.array(all_actions, dtype=np.int64))

    # Final Save
    print(f"Saving {len(all_states)} samples to {output_file}...")
    np.savez_compressed(output_file, states=np.array(all_states, dtype=np.float32), actions=np.array(all_actions, dtype=np.int64))
    print("Done!")

if __name__ == "__main__":
    generate_data()
