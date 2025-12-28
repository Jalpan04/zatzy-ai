import numpy as np
from src.game.scorecard import Category
import itertools
from collections import Counter
import math
import random

class ProExpectimaxAgent:
    def __init__(self):
        self.name = "Expectimax (Pro)"
        self.dist_cache = {}
        self._precompute_distributions()
        
        # Approximate "Future Value" of leaving a category open.
        self.category_expected_values = {
            Category.ONES: 2.1,
            Category.TWOS: 4.2,
            Category.THREES: 6.3,
            Category.FOURS: 8.4,
            Category.FIVES: 10.5,
            Category.SIXES: 12.6,
            Category.THREE_OF_A_KIND: 21.6,
            Category.FOUR_OF_A_KIND: 13.0, 
            Category.FULL_HOUSE: 22.0,     
            Category.SMALL_STRAIGHT: 25.0, 
            Category.LARGE_STRAIGHT: 32.0, 
            Category.YAHTZEE: 15.0,        
            Category.CHANCE: 22.0
        }

    def _precompute_distributions(self):
        self.dist_cache[0] = [((), 1.0)]
        for n in range(1, 6):
            outcomes = {}
            total_outcomes = 6**n
            raw_combos = itertools.combinations_with_replacement(range(1, 7), n)
            for combo in raw_combos:
                counts = Counter(combo)
                denom = 1
                for c in counts.values():
                    denom *= math.factorial(c)
                freq = math.factorial(n) // denom
                prob = freq / total_outcomes
                outcomes[combo] = prob
            self.dist_cache[n] = list(outcomes.items())

    def select_action(self, state, mask, engine=None):
        if engine is None: return ("score", 0)
        
        dice_values = tuple(sorted(engine.dice.values)) 
        rolls_left = engine.rolls_left
        scorecard = engine.scorecard
        
        if rolls_left == 0:
            best_cat = self.pick_best_category(dice_values, scorecard)
            return ("score", best_cat)
            
        best_action_type = "score"
        best_cat_immediate = self.pick_best_category(dice_values, scorecard)
        
        best_ev = self.evaluate_state(dice_values, scorecard, best_cat_immediate, final_decision=True)

        unique_keepers = set()

        # Optimization: Only check best 50% of masks if time is tight? 
        # But for 32 masks, complete search is cheap (ms).
        for keep_mask in range(32):
            kept_dice = []
            for i in range(5):
                if (keep_mask >> i) & 1:
                    kept_dice.append(dice_values[i])
            kept_dice = tuple(sorted(kept_dice))
            
            if kept_dice in unique_keepers: continue
            unique_keepers.add(kept_dice)

            n_reroll = 5 - len(kept_dice)
            
            ev = self.calculate_transition_ev(kept_dice, n_reroll, rolls_left - 1, scorecard)
            
            if ev > best_ev:
                best_ev = ev
                best_action_type = "keep"
                best_action_val = keep_mask
        
        if best_action_type == "score":
            return ("score", best_cat_immediate)
        else:
            return ("keep", best_action_val)

    def calculate_transition_ev(self, kept_dice, n_reroll, rolls_left, scorecard):
        dist = self.dist_cache[n_reroll]
        total_ev = 0.0
        
        for roll, prob in dist:
            final_hand = tuple(sorted(list(kept_dice) + list(roll)))
            
            if rolls_left == 0:
                best_cat = self.pick_best_category(final_hand, scorecard)
                val = self.evaluate_state(final_hand, scorecard, best_cat, final_decision=True)
                total_ev += prob * val
            else:
                counts = Counter(final_hand)
                if 5 in counts.values(): 
                     total_ev += prob * (50 + 100 + self.sum_future_ev(scorecard)) 
                else:
                    best_cat = self.pick_best_category(final_hand, scorecard)
                    val = self.evaluate_state(final_hand, scorecard, best_cat, final_decision=False)
                    total_ev += prob * val

        return total_ev

    def pick_best_category(self, dice, scorecard):
        available = [c for c in Category.ALL if scorecard.get_score(c) is None]
        if not available: return Category.CHANCE
        
        best_cat = available[0]
        best_val = -float('inf')
        
        for cat in available:
            val = self.evaluate_state(dice, scorecard, cat, final_decision=True)
            if val > best_val:
                best_val = val
                best_cat = cat
        return best_cat

    def sum_future_ev(self, scorecard):
        total = 0
        for cat in Category.ALL:
            if scorecard.get_score(cat) is None:
                total += self.category_expected_values.get(cat, 0)
        return total

    def evaluate_state(self, dice, scorecard, category, final_decision=False):
        """
        Calculates the Heuristic Value of picking 'category' with 'dice'.
        Uses explicit positive reinforcement for good scores.
        """
        score = scorecard.calculate_score(category, dice)
        bonus = 0
        
        # 1. Yahtzee Mechanics
        is_yahtzee_roll = (len(set(dice)) == 1)
        yahtzee_filled = (scorecard.get_score(Category.YAHTZEE) == 50)
        
        if is_yahtzee_roll:
            if category == Category.YAHTZEE:
                bonus += 500 # Massive priority
            elif yahtzee_filled:
                bonus += 100 # Joker Bonus
        
        # 2. Upper Section Strategy (Gate to 300)
        # Linear Reward for exceeding par
        if category <= Category.SIXES:
            par = category * 3
            if score >= par:
                bonus += 120 + (score - par) * 20
            else:
                # Still reward taking points if forced, but less
                bonus += score * 2 
        
        # 3. Lower Section Strategy (High Value)
        # Fixed thresholds for "Good" scores
        if category == Category.LARGE_STRAIGHT and score == 40: bonus += 60
        if category == Category.SMALL_STRAIGHT and score == 30: bonus += 40
        if category == Category.FULL_HOUSE and score == 25: bonus += 35
        if category == Category.YAHTZEE and score == 0: bonus -= 50 # Avoid zeroing Yahtzee
        
        return score + bonus
