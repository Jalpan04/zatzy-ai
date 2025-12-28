
import numpy as np
import os
import sys

# Ensure src is in path
sys.path.append(os.getcwd())

from src.game.engine import GameEngine
from src.ai.expectimax import ExpectimaxAgent

def generate_value_data(num_games=3000, output_file="value_data.npz"):
    print(f"Generating VALUE data (State -> Remaining Points) for {num_games} games...")
    
    agent = ExpectimaxAgent()
    
    all_states = []
    all_final_scores = []
    
    for game_idx in range(num_games):
        engine = GameEngine()
        game_states = []
        
        while not engine.game_over:
            # Record state at every decision point
            state = engine.get_state_vector()
            # Record current total score to calculate REMAINING points later
            current_score = engine.scorecard.get_total_score()
            game_states.append((state, current_score))
            
            # Play move
            mask = engine.get_mask()
            action_type, action_val = agent.select_action(state, mask, engine)
            engine.apply_action(action_type, action_val)
            
        # Game Over: Get Final Score
        final_score = engine.scorecard.get_total_score()
        
        # Assign (Final - Current) as the target value
        for s, cur in game_states:
            all_states.append(s)
            remaining = final_score - cur
            all_final_scores.append(remaining)
            
        if (game_idx + 1) % 50 == 0:
            print(f"Completed {game_idx + 1}/{num_games} games. Total samples: {len(all_states)}")
            np.savez_compressed(output_file, states=np.array(all_states, dtype=np.float32), values=np.array(all_final_scores, dtype=np.float32))

    print(f"Saving {len(all_states)} samples to {output_file}...")
    np.savez_compressed(output_file, states=np.array(all_states, dtype=np.float32), values=np.array(all_final_scores, dtype=np.float32))
    print("Done!")

if __name__ == "__main__":
    generate_value_data()
