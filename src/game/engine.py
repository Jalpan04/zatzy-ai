import numpy as np
from typing import List, Tuple
from .dice import Dice
from .scorecard import Scorecard, Category

class GameEngine:
    def __init__(self):
        self.dice = Dice()
        self.scorecard = Scorecard()
        self.rolls_left = 3
        self.turn_number = 1
        self.game_over = False
        
        # Start the first turn immediately
        self.start_new_turn()

    def start_new_turn(self):
        if self.scorecard.is_full():
            self.game_over = True
            return

        self.turn_number = len(self.scorecard.scores) + 1
        self.rolls_left = 3
        self.dice.reset()
        # Auto-roll the first time to give the AI something to look at
        # In this design:
        # State 0: New Turn, Dice Random, Rolls left 2.
        self.dice.roll(set()) 
        self.rolls_left -= 1
    
    def apply_action(self, action_type: str, action_value: int) -> Tuple[float, bool, bool]:
        """
        Applies an action from the AI.
        :param action_type: 'keep' or 'score'
        :param action_value: Bitmask for 'keep' (0-31) or Category ID for 'score' (1-13)
        :return: (reward, valid, game_over)
        """
        if self.game_over:
            return 0, False, True

        if action_type == 'keep':
            if self.rolls_left <= 0:
                return -10, False, self.game_over # Illegal to roll with 0 rolls left
            
            # Decode bitmask
            # 5 bits: 00000 to 11111. 1 = Keep, 0 = Re-roll.
            # However, implementation of dice.roll takes INDICES of kept dice.
            keep_indices = set()
            for i in range(5):
                if (action_value >> i) & 1:
                    keep_indices.add(i)
            
            self.dice.roll(keep_indices)
            self.rolls_left -= 1
            return 0, True, False

        elif action_type == 'score':
            category = action_value
            try:
                score = self.scorecard.score(category, self.dice.get_values())
                # Turn End
                self.start_new_turn()
                return score, True, self.game_over
            except ValueError:
                # Category already full
                return -10, False, self.game_over
        
        return 0, False, self.game_over

    def get_state_vector(self) -> np.ndarray:
        """
        Returns the encoded state vector for the AI.
        Size: 5 (Dice) + 13 (Scorecard Flags) + 1 (Upper Score Norm) + 1 (Yahtzee Bonus) + 1 (Rolls Left Norm) = 21 floats.
        Optimization: Dice could be one-hot, but starting with normalized floats (val/6).
        """
        # 1. Dice (Sorted, Normalized 0-1)
        sorted_dice = self.dice.get_sorted_values()
        dice_vec = [d / 6.0 for d in sorted_dice]

        # 1b. Dice Histogram (Counts of 1s, 2s, ... 6s) - NEW FEATURE
        # 1b. Dice Histogram (Counts of 1s, 2s, ... 6s)
        from collections import Counter
        counts = Counter(sorted_dice)
        # Normalized by 5 (max count)
        hist_vec = [counts[i] / 5.0 for i in range(1, 7)]

        # 2. Scorecard Flags (0 = Empty, 1 = Filled)
        scorecard_flags = []
        for cat in Category.ALL:
            if self.scorecard.get_score(cat) is None:
                scorecard_flags.append(0.0)
            else:
                scorecard_flags.append(1.0)
        
        # --- GOD VIEW FEATURES (NEW) ---
        
        # 6. Potential Scores (Normalized)
        # We ask the scorecard: "If I picked this category now, what would I get?"
        # We normalize by 50 (max meaningful single score, excluding bonuses)
        potential_scores = []
        for cat in Category.ALL:
            # We use the raw dice values
            score = self.scorecard.calculate_score(cat, self.dice.get_values())
            potential_scores.append(min(score / 50.0, 1.0))

        # 7. Explicit Pattern Flags
        # Help the NN see patterns without arithmetic
        counts_vals = list(counts.values())
        has_pair = 1.0 if any(c >= 2 for c in counts_vals) else 0.0
        has_two_pairs = 1.0 if len([c for c in counts_vals if c >= 2]) >= 2 else 0.0
        has_triplet = 1.0 if any(c >= 3 for c in counts_vals) else 0.0
        has_quad = 1.0 if any(c >= 4 for c in counts_vals) else 0.0
        has_yahtzee_roll = 1.0 if any(c >= 5 for c in counts_vals) else 0.0
        
        # Full House (3 and 2, or 5)
        has_full_house = 1.0 if (has_triplet and has_pair and len(counts) == 2) or has_yahtzee_roll else 0.0
        
        # Straights
        unique_dice = sorted(list(set(sorted_dice)))
        s_set = set(unique_dice)
        has_sm_straight = 1.0 if ({1,2,3,4}.issubset(s_set) or {2,3,4,5}.issubset(s_set) or {3,4,5,6}.issubset(s_set)) else 0.0
        has_lg_straight = 1.0 if ({1,2,3,4,5}.issubset(s_set) or {2,3,4,5,6}.issubset(s_set)) else 0.0
        
        pattern_flags = [has_pair, has_two_pairs, has_triplet, has_quad, has_full_house, has_sm_straight, has_lg_straight, has_yahtzee_roll]

        # 3. Upper Score (Normalized by 63)
        upper_score = 0
        for cat in Category.UPPER:
            val = self.scorecard.get_score(cat)
            if val is not None:
                upper_score += val
        upper_norm = min(upper_score / 63.0, 1.0) 

        # 4. Yahtzee Bonus
        yahtzee_secured = 1.0 if (self.scorecard.get_score(Category.YAHTZEE) == 50) else 0.0

        # 5. Rolls Left
        rolls_norm = self.rolls_left / 2.0 

        # Total Size breakdown:
        # Dice(5) + Hist(6) + Flags(13) + Potentials(13) + Patterns(8) + Context(3) = 48
        return np.array(dice_vec + hist_vec + scorecard_flags + potential_scores + pattern_flags + [upper_norm, yahtzee_secured, rolls_norm], dtype=np.float32)

    def get_mask(self) -> np.ndarray:
        """
        Returns a mask of valid actions.
        Output Vector Size: 32 (Keep) + 13 (Score) = 45.
        1.0 = Valid, 0.0 (or -inf) = Invalid.
        """
        # Keep Actions (0-31): Valid only if Rolls Left > 0
        keep_mask = np.ones(32, dtype=np.float32)
        if self.rolls_left == 0:
            keep_mask[:] = 0.0 # Can't roll anymore

        # Score Actions (0-12 representing Categories 1-13)
        score_mask = np.zeros(13, dtype=np.float32)
        for i, cat in enumerate(Category.ALL):
            if self.scorecard.get_score(cat) is None:
                score_mask[i] = 1.0 # Available
        
        return np.concatenate([keep_mask, score_mask])
