
import numpy as np
import torch
import sys, os
sys.path.append(os.getcwd())

from src.game.engine import GameEngine
from src.ai.neuro_expectimax import NeuroExpectimaxAgent
from src.ai.expectimax import ExpectimaxAgent

def run_benchmark(num_games=50):
    print(f"Benchmarking Neuro-Expectimax over {num_games} games...")
    
    # Neuro Agent
    neuro_agent = NeuroExpectimaxAgent()
    # Baseline Agent
    baseline_agent = ExpectimaxAgent()
    
    neuro_scores = []
    baseline_scores = []
    
    for i in range(num_games):
        # Neuro Game
        e_neuro = GameEngine()
        while not e_neuro.game_over:
            state = e_neuro.get_state_vector()
            mask = e_neuro.get_mask()
            action_type, action_val = neuro_agent.select_action(state, mask, e_neuro)
            e_neuro.apply_action(action_type, action_val)
        neuro_scores.append(e_neuro.scorecard.get_total_score())
        
        # Baseline Game
        e_base = GameEngine()
        while not e_base.game_over:
            state = e_base.get_state_vector()
            mask = e_base.get_mask()
            action_type, action_val = baseline_agent.select_action(state, mask, e_base)
            e_base.apply_action(action_type, action_val)
        baseline_scores.append(e_base.scorecard.get_total_score())
        
        if (i+1) % 5 == 0:
            print(f"Completed {i+1}/{num_games} games...")
            print(f"  Neuro Avg: {np.mean(neuro_scores):.1f} | Max: {np.max(neuro_scores)} | Last: {neuro_scores[-1]}")
            print(f"  Base Avg:  {np.mean(baseline_scores):.1f} | Max: {np.max(baseline_scores)} | Last: {baseline_scores[-1]}")
        else:
            print(f"Game {i+1}/{num_games} - Neuro: {neuro_scores[-1]}, Base: {baseline_scores[-1]}")
            
    print("\n--- FINAL BENCHMARK RESULTS ---")
    print(f"Neuro-Expectimax: Avg={np.mean(neuro_scores):.1f}, Max={np.max(neuro_scores)}, Min={np.min(neuro_scores)}")
    print(f"Expectimax Base:  Avg={np.mean(baseline_scores):.1f}, Max={np.max(baseline_scores)}, Min={np.min(baseline_scores)}")

if __name__ == "__main__":
    try:
        run_benchmark(num_games=50)
    except Exception as e:
        import traceback
        traceback.print_exc()
