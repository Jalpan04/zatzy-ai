import numpy as np
from src.game.scorecard import Category
import itertools

class ExpectimaxAgent:
    def __init__(self):
        self.name = "Expectimax (Math)"
        
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
        # 0 to 31
        for keep_mask in range(32):
            # Parse mask
            kept_dice = []
            for i in range(5):
                if (keep_mask >> i) & 1:
                    kept_dice.append(dice_values[i])
            
            n_reroll = 5 - len(kept_dice)
            
            # Calculate EV of this Keep decision
            # EV = Sum(Prob(outcome) * Value(outcome))
            # Value(outcome) = Max(Score Now, EV of Next Roll)
            
            ev = self.calculate_keep_ev(kept_dice, n_reroll, rolls_left - 1, scorecard)
            
            if ev > best_ev:
                best_ev = ev
                best_action_type = "keep"
                best_action_val = keep_mask
                
        return (best_action_type, best_action_val)

    def calculate_keep_ev(self, kept_dice, n_reroll, rolls_left, scorecard):
        # Base case: Known outcomes of rolling N dice
        # We can iterate all 6^n outcomes. For n=5, 7776 outcomes. Doable.
        
        outcomes = itertools.product(range(1, 7), repeat=n_reroll)
        total_ev = 0
        count = 0
        
        for roll in outcomes:
            final_hand = kept_dice + list(roll)
            count += 1
            
            if rolls_left == 0:
                # Terminal node: Value is best score achievable
                best_cat = self.pick_best_category(final_hand, scorecard)
                total_ev += self.score_heuristic(best_cat, final_hand, scorecard)
            else:
                # Recursive step? Too slow.
                # Heuristic: Approximate next layer by just taking max score potential
                # or simplified lookahead.
                # For speed in Python, we limit depth or rely on heuristic 'hand value'
                best_cat = self.pick_best_category(final_hand, scorecard)
                score_val = self.score_heuristic(best_cat, final_hand, scorecard)
                
                # Bonus for "Potential" (e.g. nearly straight, lots of 6s)
                # This approximates the future rolls without full recursion
                potential_bonus = 0
                if rolls_left > 0:
                     # Very rough heuristic for 'potential' of a hand
                     # e.g. 4 of a kind has high potential to become Yahtzee
                     counts = {x: final_hand.count(x) for x in set(final_hand)}
                     if 5 not in counts.values() and any(c >= 3 for c in counts.values()):
                         potential_bonus += 10 # Chance for Yahtzee
                     
                total_ev += max(score_val, score_val + potential_bonus) # Simple logic

        return total_ev / count

    def pick_best_category(self, dice, scorecard):
        best_cat = -1
        best_val = -1.0
        available = [c for c in Category.ALL if scorecard.get_score(c) is None]
        for cat in available:
            val = self.score_heuristic(cat, dice, scorecard)
            if val > best_val:
                best_val = val
                best_cat = cat
        return best_cat

    def score_heuristic(self, cat, dice, scorecard):
        # Returns the "Value" of scoring this category
        # Score + Strategic Weight
        score = scorecard.calculate_score(cat, dice)
        
        # Strategic Weights
        weight = 0
        
        # Upper Bonus Strategy: 
        # Points in Upper section are worth MORE if they help reach 63
        if cat <= Category.SIXES:
            # Normalized value: 3 of a kind is 'par'
            # e.g. rolling three 6s (18) is good. Rolling one 6 (6) is bad for bonus.
            par = (cat + 1) * 3
            diff = score - par
            weight += diff * 0.5 # Encourage slightly
            
        # Yahtzee is critical
        if cat == Category.YAHTZEE and score == 50:
            weight += 100 # Huge priority
            
        return score + weight
