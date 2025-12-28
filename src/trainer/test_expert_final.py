import sys, os
sys.path.append(os.getcwd())
from src.game.engine import GameEngine
from src.ai.expectimax import ExpectimaxAgent
import numpy as np

def benchmark(num_games=10):
    agent = ExpectimaxAgent()
    scores = []
    print(f"Benchmarking {agent.name} (God-Tier Heuristic) for {num_games} games...")
    
    for i in range(num_games):
        engine = GameEngine()
        while not engine.game_over:
            print(f"\rGame {i+1} Progress: Turn {engine.turn_number}/13", end="")
            state = engine.get_state_vector()
            mask = engine.get_mask()
            action_type, action_val = agent.select_action(state, mask, engine)
            engine.apply_action(action_type, action_val)
            
        score = engine.scorecard.get_total_score()
        scores.append(score)
        print(f"\rGame {i+1}: {score}                        ")
        
    print(f"\nAverage: {np.mean(scores)}")
    print(f"Max: {np.max(scores)}")
    print(f"Min: {np.min(scores)}")

if __name__ == "__main__":
    benchmark(10)
