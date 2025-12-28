
import numpy as np
import sys, os
sys.path.append(os.getcwd())

from src.game.engine import GameEngine
from src.ai.expectimax import ExpectimaxAgent

def test_expert():
    agent = ExpectimaxAgent()
    scores = []
    print("Testing Superhuman Expert (10 games)...")
    for i in range(10):
        engine = GameEngine()
        while not engine.game_over:
            state = engine.get_state_vector()
            mask = engine.get_mask()
            action_type, action_val = agent.select_action(state, mask, engine)
            engine.apply_action(action_type, action_val)
        scores.append(engine.scorecard.get_total_score())
        print(f"Game {i+1}: {scores[-1]}")
    
    print(f"\nAvg: {np.mean(scores)}")
    print(f"Max: {np.max(scores)}")

if __name__ == "__main__":
    test_expert()
