from src.game.engine import GameEngine
from src.ai.agent import Agent
import numpy as np

class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, agent: Agent, num_games=1) -> float:
        """
        Runs num_games for the agent and returns the average score.
        """
        total_score = 0
        for _ in range(num_games):
            engine = GameEngine()
            
            # Game Loop
            # Limit turns just in case (though engine handles it)
            max_steps = 1000 
            steps = 0
            
            while not engine.game_over and steps < max_steps:
                state = engine.get_state_vector()
                mask = engine.get_mask()
                
                action_type, action_val = agent.select_action(state, mask)
                
                # Apply Action
                # Reward is purely points scored this turn (or penalty)
                points, valid, game_over = engine.apply_action(action_type, action_val)
                
                if not valid:
                    # Should not happen with correct masking logic
                    # Penalize heavily if it does?
                    # For now, just break loop or ignore.
                    # With masking, argmax shouldn't pick invalid unless all are invalid (bug).
                    break
                
                steps += 1
            
            total_score += engine.scorecard.get_total_score()
            
        return total_score / num_games
