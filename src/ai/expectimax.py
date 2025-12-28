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
        # Use precomputed distribution
        # distribution is list of (roll_tuple, probability)
        dist = self.dist_cache[n_reroll]
        
        total_ev = 0
        
        for roll, prob in dist:
            final_hand = kept_dice + list(roll)
            
            if rolls_left == 0:
                # Terminal node
                best_cat = self.pick_best_category(final_hand, scorecard)
                score_val = self.score_heuristic(best_cat, final_hand, scorecard)
                total_ev += prob * score_val
            else:
                # Heuristic Lookahead (Approximation)
                # Instead of full recursion (which would be slow even with init optimizations),
                # we assume the player takes the best 'potential' score available.
                best_cat = self.pick_best_category(final_hand, scorecard)
                score_val = self.score_heuristic(best_cat, final_hand, scorecard)
                
                # Simple potential bonus for being closer to Yahtzee/Straight
                # (This can be tuned or replaced with ML distillation!)
                potential_bonus = 0
                counts = Counter(final_hand)
                if 5 not in counts.values(): # Not already Yahtzee
                    max_count = max(counts.values()) if counts else 0
                    if max_count >= 3: potential_bonus += 15 # 3 or 4 of kind
                    if max_count >= 4: potential_bonus += 25
                
                # Check straights roughly
                uniques = len(set(final_hand))
                if uniques >= 4: potential_bonus += 10
                
                total_ev += prob * (score_val + potential_bonus)

        return total_ev

    def pick_best_category(self, dice, scorecard):
        best_cat = -1
        best_val = -float('inf')
        # Optimized: Only check available
        available = [c for c in Category.ALL if scorecard.get_score(c) is None]
        if not available: return Category.CHANCE # Fallback
        
        for cat in available:
            val = self.score_heuristic(cat, dice, scorecard)
            if val > best_val:
                best_val = val
                best_cat = cat
        return best_cat

    def score_heuristic(self, cat, dice, scorecard):
        score = scorecard.calculate_score(cat, dice)
        weight = 0
        
        # Upper Bonus Strategy
        if cat <= Category.SIXES:
            # Normalized value logic
            val = (cat) # 1-6 map? No, enum is 1-6 for ONES..SIXES?
            # Check Category definition. 
            # In Category class: ONES=1...SIXES=6.
            # Dice values are 1-6.
            # A score of 3*val is Par.
            # Difference from Par is valuable.
            if cat in [Category.ONES, Category.TWOS, Category.THREES, Category.FOURS, Category.FIVES, Category.SIXES]:
                 # Assuming cat matches dice face value directly for 1-6
                 # Careful: Category.ONES might be 1, but dice face is 1.
                 # Category.NAME_MAP keys are integers.
                 # Let's assume standard mapping: ONES (1) scores sums of 1s.
                 # Par = 3 * cat (if cat 1..6 maps to faces 1..6)
                 # Wait, looking at src/game/scorecard.py...
                 # Category.ONES = 1. Category.SIXES = 6.
                 # So yes, face_value = cat.
                 
                 par = cat * 3
                 diff = score - par
                 weight += diff * 2.0 # Higher weight for upper bonus
        
        # Yahtzee Strategy
        if cat == Category.YAHTZEE:
            if score == 50: weight += 200 # Massive priority
            elif score == 0: weight -= 100 # Avoid zeroing Yahtzee unless necessary
            
        return score + weight
