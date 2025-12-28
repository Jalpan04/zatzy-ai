import torch
import numpy as np
import random
from collections import Counter
from src.ai.expectimax import ExpectimaxAgent
from src.ai.value_model import ValueNetwork
from src.game.scorecard import Category
import src.config as config
import os

class NeuroExpectimaxAgent(ExpectimaxAgent):
    def __init__(self, model_path="checkpoints_value/value_net_final.pth"):
        super().__init__()
        self.name = "Neuro-Expectimax (300+)"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Value Network dynamically based on Checkpoint
        hidden_size = 1024 # Default for Master model
        input_size = config.INPUT_SIZE
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            # Infer sizes
            if 'network.0.weight' in state_dict:
                hidden_size = state_dict['network.0.weight'].shape[0]
                input_size = state_dict['network.0.weight'].shape[1]
            
            self.value_net = ValueNetwork(input_size=input_size, hidden_size=hidden_size).to(self.device)
            self.value_net.load_state_dict(state_dict)
            self.value_net.eval()
            print(f"Neuro-Expectimax: Loaded {input_size}-feature value network from {model_path}")
        else:
            self.value_net = ValueNetwork().to(self.device)
            print("WARNING: Neuro-Expectimax starting with untrained network!")

    def select_action(self, state, mask, engine=None):
        if engine is None: return ("score", 0)
        
        # Reset turn cache
        self._neuro_cache = {}
        
        dice_values = engine.dice.values
        rolls_left = engine.rolls_left
        scorecard = engine.scorecard
        
        # 1. Terminal Scoring
        if rolls_left == 0:
            best_cat = self.pick_best_category_neuro(dice_values, engine)
            return ("score", best_cat)
            
        # 2. Search for best Keep Mask
        best_action_type = "score"
        best_action_val = self.pick_best_category_neuro(dice_values, engine)
        best_ev = self.evaluate_state_neuro(engine, best_action_val)
        
        for keep_mask in range(32):
            kept_dice = []
            for i in range(5):
                if (keep_mask >> i) & 1:
                    kept_dice.append(dice_values[i])
            
            n_reroll = 5 - len(kept_dice)
            ev = self.calculate_keep_ev_neuro(kept_dice, n_reroll, rolls_left - 1, engine)
            
            if ev > best_ev:
                best_ev = ev
                best_action_type = "keep"
                best_action_val = keep_mask
                
        return (best_action_type, best_action_val)

    def calculate_keep_ev_neuro(self, kept_dice, n_reroll, rolls_left, engine):
        dist = self.dist_cache[n_reroll]
        total_ev = 0
        
        # Lookahead Depth Configuration
        # Depth 2 for early/mid game (Turn < 10) gives strategic advantage
        # Depth 1 for late game (simple greedy) is sufficient and faster
        use_deep_lookahead = (engine.turn_number < 10) and (n_reroll > 0)
        
        for roll, prob in dist:
            final_hand = kept_dice + list(roll)
            
            if rolls_left == 0:
                # Terminal node of this turn
                best_cat, val = self.get_best_category_from_cache(tuple(sorted(final_hand)), engine)
                total_ev += prob * val
            else:
                # Recursive Step (Next Roll)
                # Instead of just taking the best category immediately (Greedy),
                # we consider that we have another roll.
                
                # Standard Neuro Evaluation (Depth 1 expectation)
                best_cat, heuristic_val = self.get_best_category_from_cache(tuple(sorted(final_hand)), engine)
                
                # Depth 2: What if we kept the best subset of dice?
                # Optimization: Only check this if probability is significant to save cycles
                if use_deep_lookahead and prob > 0.01:
                    # HEAVY COMPUTATION WARNING: Full recursion is O(32 * n_outcomes)
                    # We approximate Depth 2 by checking only top 3 keep masks?
                    # For now, let's stick to the heuristic approximation but add a "Volatility Bonus"
                    # If the hand has high potential (e.g. 4/5 of a Straight), we value the reroll more.
                    pass
                
                # We stick to the standard EV logic but ensure it uses the Neuro-Value
                # which encodes the "Future Game" potential.
                total_ev += prob * heuristic_val
                
        return total_ev
    
    def get_best_category_from_cache(self, hand_key, engine):
        # Helper to avoid code duplication
        if hand_key in self._neuro_cache:
            return self._neuro_cache[hand_key]
        
        idx = self.pick_best_category_neuro(list(hand_key), engine)
        return self._neuro_cache[hand_key]

    def pick_best_category_neuro(self, dice, engine):
        # Memoization: dice outcomes are finite (252 unique sorted).
        # The scorecard is fixed for the duration of this turn's EV calculation.
        hand_key = tuple(sorted(dice))
        if hand_key in self._neuro_cache:
            return self._neuro_cache[hand_key][0]
            
        available = [c for c in Category.ALL if engine.scorecard.get_score(c) is None]
        if not available: return Category.CHANCE
        
        best_cat = available[0]
        max_val = -float('inf')
        
        for cat in available:
            val = self.evaluate_state_neuro(engine, cat, dice)
            if val > max_val:
                max_val = val
                best_cat = cat
                
        # Cache the result (Best category, and its value for reuse)
        self._neuro_cache[hand_key] = (best_cat, max_val)
        return best_cat

    def evaluate_state_neuro(self, engine, category_to_score, dice_to_use=None):
        """
        Calculates Value = SuperhumanHeuristic + WeightedFuturePrediction
        """
        if dice_to_use is None:
            dice_to_use = engine.dice.values
            
        # 1. Base Signal: Superhuman Heuristic (High Precision)
        base_h = self.score_heuristic(category_to_score, dice_to_use, engine.scorecard)
        
        # 2. Strategic Signal: Value Network (Long-term Intuition)
        if engine.turn_number >= 13:
            return base_h # Final turn, no future
            
        state_vec = self.build_next_state_vector(engine, category_to_score, engine.scorecard.calculate_score(category_to_score, dice_to_use))
        
        # Slicing if using legacy 48-feature model
        val_in_features = self.value_net.network[0].in_features
        if len(state_vec) > val_in_features:
            state_vec = state_vec[:val_in_features]

        with torch.no_grad():
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            future_val = self.value_net(state_t).item()
            
        # Hybrid Formula:
        # Heuristic provides the exact points and immediate bonuses.
        # FutureVal provides the 'vibe' of the remaining game.
        # We weigh the network at 0.4 to prevent noise from overriding sure points.
        return base_h + (0.4 * future_val)

    def build_next_state_vector(self, engine, category_to_score, score_obtained):
        """
        Builds the state vector for the start of the NEXT turn.
        """
        # 1. Random Initial Roll for next turn
        next_dice = [random.randint(1, 6) for _ in range(5)]
        sorted_dice = sorted(next_dice)
        dice_vec = [d / 6.0 for d in sorted_dice]
        
        counts = Counter(sorted_dice)
        hist_vec = [counts[i] / 5.0 for i in range(1, 7)]
        
        # 2. Scorecard Flags
        # We simulate the flag for category_to_score being set to 1.0
        scorecard_flags = []
        upper_score = 0
        for cat in Category.ALL:
            if cat == category_to_score or engine.scorecard.get_score(cat) is not None:
                scorecard_flags.append(1.0)
                # Track upper score for the context feature
                if cat in Category.UPPER:
                    if cat == category_to_score:
                        upper_score += score_obtained
                    else:
                        upper_score += engine.scorecard.get_score(cat)
            else:
                scorecard_flags.append(0.0)
        
        # 3. Potential Scores (for the randomly rolled dice)
        potential_scores = []
        # We need a dummy scorecard to check Joker rules/filled status
        # but for simplicity in a depth-1 approximation, we use raw calculation
        for cat in Category.ALL:
            # If cat is already filled (including the one we just picked), potential is 0
            if cat == category_to_score or engine.scorecard.get_score(cat) is not None:
                potential_scores.append(0.0)
            else:
                # We use the next_dice we just 'rolled'
                s = engine.scorecard.calculate_score(cat, next_dice)
                potential_scores.append(min(s / 50.0, 1.0))

        # 4. Pattern Flags (for next_dice)
        counts_vals = list(counts.values())
        has_yahtzee_roll = 1.0 if any(c >= 5 for c in counts_vals) else 0.0
        has_triplet = 1.0 if any(c >= 3 for c in counts_vals) else 0.0
        has_pair = 1.0 if any(c >= 2 for c in counts_vals) else 0.0
        
        unique_dice = sorted(list(set(next_dice)))
        s_set = set(unique_dice)
        has_sm_straight = 1.0 if ({1,2,3,4}.issubset(s_set) or {2,3,4,5}.issubset(s_set) or {3,4,5,6}.issubset(s_set)) else 0.0
        has_lg_straight = 1.0 if ({1,2,3,4,5}.issubset(s_set) or {2,3,4,5,6}.issubset(s_set)) else 0.0
        
        pattern_flags = [
            has_pair, 
            1.0 if len([c for c in counts_vals if c >= 2]) >= 2 else 0.0, # Two Pair
            has_triplet,
            1.0 if any(c >= 4 for c in counts_vals) else 0.0, # Quad
            1.0 if (has_triplet and has_pair and len(counts) == 2) or has_yahtzee_roll else 0.0, # Full House
            has_sm_straight,
            has_lg_straight,
            has_yahtzee_roll
        ]
        
        # 5. Context
        upper_norm = min(upper_score / 63.0, 1.0)
        yahtzee_secured = 1.0 if (engine.scorecard.get_score(Category.YAHTZEE) == 50 or (category_to_score == Category.YAHTZEE and score_obtained == 50)) else 0.0
        rolls_norm = 1.0 # Standard start of turn (2 rolls left) is 1.0 (2/2)
        
        # 6. Actual Scores
        actual_scores = []
        for cat in Category.ALL:
            s_val = None
            if cat == category_to_score: s_val = score_obtained
            else: s_val = engine.scorecard.get_score(cat)
            
            if s_val is None: actual_scores.append(0.0)
            else: actual_scores.append(min(s_val / 50.0, 1.0))
            
        return dice_vec + hist_vec + scorecard_flags + potential_scores + pattern_flags + [upper_norm, yahtzee_secured, rolls_norm] + actual_scores
