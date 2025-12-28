import numpy as np
from src.game.scorecard import Category
import itertools
from collections import Counter
import math

class ExpectimaxAgent:
    def __init__(self):
        self.name = "Expectimax (Math)"
        self.dist_cache = {} # Cache for dice distributions
        self._precompute_distributions()
        
    def _precompute_distributions(self):
        # Precompute outcome probabilities for rolling n dice (1 to 5)
        # Instead of 6^n loop, we iterate unique sorted outcomes
        # and multiply by multinomial probability.
        
        for n in range(1, 6):
            outcomes = {} # {sorted_tuple: probability}
            total_outcomes = 6**n
            
            # Generate all combinations (sorted)
            raw_combos = itertools.combinations_with_replacement(range(1, 7), n)
            
            for combo in raw_combos:
                # Calculate frequency of this combination (Multinomial calc)
                # Count = n! / (c1! * c2! * ... * c6!)
                counts = Counter(combo)
                denom = 1
                for c in counts.values():
                    denom *= math.factorial(c)
                freq = math.factorial(n) // denom
                
                prob = freq / total_outcomes
                outcomes[combo] = prob
            
            self.dist_cache[n] = list(outcomes.items())
        
        # n=0 case
        self.dist_cache[0] = [((), 1.0)]

    def select_action(self, state, mask, engine=None):
        if engine is None: return ("score", 0)
        
        dice_values = engine.dice.values
        rolls_left = engine.rolls_left
        scorecard = engine.scorecard
        
        # 1. Always Score if no rolls left
        if rolls_left == 0:
            best_cat = self.pick_best_category(dice_values, scorecard)
            return ("score", best_cat)
            
        # 2. Compare Expected Value of Rolling vs Scoring Immediately
        best_action_type = "score"
        best_action_val = self.pick_best_category(dice_values, scorecard)
        best_ev = self.score_heuristic(best_action_val, dice_values, scorecard)
        
        # Check all 32 Keep Combinations
        for keep_mask in range(32):
            kept_dice = []
            for i in range(5):
                if (keep_mask >> i) & 1:
                    kept_dice.append(dice_values[i])
            
            n_reroll = 5 - len(kept_dice)
            
            ev = self.calculate_keep_ev(kept_dice, n_reroll, rolls_left - 1, scorecard)
            
            if ev > best_ev:
                best_ev = ev
                best_action_type = "keep"
                best_action_val = keep_mask
                
        return (best_action_type, best_action_val)

    def calculate_keep_ev(self, kept_dice, n_reroll, rolls_left, scorecard):
        # We assume n_reroll, rolls_left fixed for this call
        dist = self.dist_cache[n_reroll]
        total_ev = 0
        
        for roll, prob in dist:
            final_hand = kept_dice + list(roll)
            
            if rolls_left == 0:
                best_cat = self.pick_best_category(final_hand, scorecard)
                score_val = self.score_heuristic(best_cat, final_hand, scorecard)
                total_ev += prob * score_val
            else:
                # Heuristic Lookahead (Approximation)
                best_cat = self.pick_best_category(final_hand, scorecard)
                score_val = self.score_heuristic(best_cat, final_hand, scorecard)
                
                # Pro-Level Potential Bonus
                pot_bonus = 0
                counts = Counter(final_hand).values()
                mx = max(counts) if counts else 0
                
                # 1. Yahtzee Hunt (Only if open)
                if scorecard.get_score(Category.YAHTZEE) is None:
                    if mx == 3: pot_bonus += 20
                    if mx == 4: pot_bonus += 60
                
                # 2. Upper Bonus Tracker
                # We calculate if this hand helps reach the 'Par' target.
                
                total_ev += prob * (score_val + pot_bonus)

        return total_ev

    def pick_best_category(self, dice, scorecard):
        available = [c for c in Category.ALL if scorecard.get_score(c) is None]
        if not available: return Category.CHANCE
        
        best_cat = available[0]
        best_val = -float('inf')
        
        for cat in available:
            val = self.score_heuristic(cat, dice, scorecard)
            if val > best_val:
                best_val = val
                best_cat = cat
        return best_cat

    def score_heuristic(self, cat, dice, scorecard):
        score = scorecard.calculate_score(cat, dice)
        bonus = 0
        
        # 1. Upper Section Strategy (Gate to 300)
        # Linear scaling: Reward every point above/below Par
        if cat <= Category.SIXES:
            par = cat * 3
            if score >= par:
                # High rewards for points at or above par
                bonus += 120 + (score - par) * 20  # Explicit reward for surplus
            else:
                # Small reward for sub-par, but less than par
                # Prevents "Sacrificial Scoring" but keeps AI hunting for 3+
                # e.g. score=2 (on 4s). Par=12. Better than 0.
                bonus += score * 3
        
        # 2. Yahtzee Strategy (God-Tier Priority)
        if cat == Category.YAHTZEE:
            if score == 50: bonus += 500 # Massive priority (was 300)
            else: bonus -= 50 # Only light penalty to avoid zeroing if possible
            
        # 3. Straights and High Value Lowers
        # Bonuses help distinguish "bad" trash slots from "good" trash slots
        if cat == Category.LARGE_STRAIGHT and score == 40: bonus += 60 # Was 80, reduced slightly to prioritize Upper
        if cat == Category.SMALL_STRAIGHT and score == 30: bonus += 40
        if cat == Category.FULL_HOUSE and score == 25: bonus += 35 
            
        return score + bonus
