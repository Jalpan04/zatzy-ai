import random
import math
import copy

class MCTSAgent:
    def __init__(self, simulations=50):
        self.name = "MCTS (Simulation)"
        self.simulations = simulations
        
    def select_action(self, state, mask, engine):
        """
        Simplified MCTS (Flat Monte Carlo / Pure Monte Carlo Search).
        For each legal move, play valid random games to the end.
        Pick the move with highest average final score.
        """
        legal_actions = []
        
        # 1. Enumerate Legal Actions
        if engine.rolls_left > 0:
            # Keep Actions (0-31)
            # Optimization: Only check masks that match current dice?
            # Or just check all 32 and filter?
            # engine.get_mask() gives us validity.
            for i in range(32):
                if mask[i]:
                    legal_actions.append(('keep', i))
        else:
            # Score Actions (0-12)
            for i in range(13):
                # Mask index 32+i corresponds to Category i+1
                if mask[32+i]:
                    legal_actions.append(('score', i+1))
                    
        # 2. Run Simulations
        best_avg_score = -1
        best_action = None
        
        # Limit actions to check if too many?
        # For 'keep', 32 is manageable.
        
        for action in legal_actions:
            total_score = 0
            # Run N simulations for this candidate action
            for _ in range(self.simulations):
                sim_score = self.simulate(engine, action)
                total_score += sim_score
            
            avg_score = total_score / self.simulations
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_action = action
                
        return best_action

    def simulate(self, real_engine, first_action):
        """
        Clones the engine, applies the first action, then plays randomly to end.
        """
        # We need a lightweight clone or we simulate manually?
        # Cloning the full engine is slow.
        # But for MCTS correctness, we must tracking scorecard state.
        
        # Fast Clone (Manual)
        sim_engine = copy.deepcopy(real_engine)
        
        # Apply the candidate move
        sim_engine.apply_action(first_action[0], first_action[1])
        
        # Play randomly until game over
        while not sim_engine.game_over:
            # Random Move
            mask = sim_engine.get_mask()
            
            # Weighted Random? Or Pure Random? Pure Random is standard MCTS rollout.
            # But Pure Random Yahtzee is terrible (Avg 60). This MCTS will likely be bad without a "Heuristic Rollout".
            # Let's use a "Sensible Random" (Greedy Selection for scoring, Random for keeping).
            
            if sim_engine.rolls_left == 0:
                # Must score. Pick Best available points (Greedy Rollout)
                # This makes it "MCTS with Heuristic Rollout"
                best_cat_val = -1
                best_cat_idx = None
                
                # Check valid categories
                for i in range(13):
                    if mask[32+i]:
                        # Calculate score
                        cat_idx = i + 1
                        points = sim_engine.scorecard.calculate_score(cat_idx, sim_engine.dice.values)
                        if points > best_cat_val:
                             best_cat_val = points
                             best_cat_idx = cat_idx
                             
                sim_engine.apply_action('score', best_cat_idx)
            else:
                # Random Keep
                # Pick a random valid keep mask
                # Optimization: Bias towards keeping high dice? 
                # Let's stick to uniform random for 0-31 to be "Base MCTS".
                valid_keeps = [i for i in range(32) if mask[i]]
                choice = random.choice(valid_keeps)
                sim_engine.apply_action('keep', choice)
                
        return sim_engine.scorecard.get_total_score()
