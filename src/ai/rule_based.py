from src.game.scorecard import Category
from src.game.dice import Dice

class RuleBasedAgent:
    def __init__(self):
        self.name = "Rule-Based (Expert)"
        
    def select_action(self, state, mask, engine):
        """
        Decision Logic:
        1. Always Keep Yahtzees.
        2. Keep Large Straights if 4/5 dice match.
        3. Prioritize Upper Section (specifically 4s, 5s, 6s).
        """
        dice_values = sorted(engine.dice.values)
        counts = {x: dice_values.count(x) for x in set(dice_values)}
        
        # 1. Action: Keeping Dice (Roll 1 & 2)
        if engine.rolls_left > 0:
            # Rule 1: Keep 3+ of a kind
            for val, count in counts.items():
                if count >= 3:
                     # Keep all of these
                     # Mask logic: which indices have this val?
                     keep_mask = 0
                     for i, d in enumerate(engine.dice.values):
                         if d == val:
                             keep_mask |= (1 << i)
                     return 'keep', keep_mask
            
            # Rule 2: Keep 4-run (Straight hunt)
            # Simplified: If we have 1,2,3,4 or 2,3,4,5 or 3,4,5,6
            # Keep those.
            
            # Fallback: Keep 4s, 5s, 6s (High Value)
            keep_mask = 0
            for i, d in enumerate(engine.dice.values):
                if d >= 4:
                    keep_mask |= (1 << i)
            
            return 'keep', keep_mask

        # 2. Action: Scoring (Roll 3)
        else:
            # Best available category
            scorecard = engine.scorecard
            best_cat = Category.CHANCE
            best_score = -1
            
            # Priority List approach
            priorities = [
                Category.YAHTZEE, 
                Category.LARGE_STRAIGHT, 
                Category.SIXES, Category.FIVES, Category.FOURS,
                Category.FULL_HOUSE, 
                Category.SMALL_STRAIGHT,
                Category.THREES, Category.TWOS, Category.ONES,
                Category.THREE_OF_A_KIND, Category.FOUR_OF_A_KIND,
                Category.CHANCE
            ]
            
            for cat in priorities:
                if scorecard.get_score(cat) is None:
                    # Check if decent score?
                    score = scorecard.calculate_score(cat, dice_values)
                    
                    # Acceptance Thresholds
                    if cat == Category.YAHTZEE and score == 50: return 'score', cat
                    if cat == Category.LARGE_STRAIGHT and score == 40: return 'score', cat
                    if cat == Category.FULL_HOUSE and score == 25: return 'score', cat
                    if cat >= Category.ONES and cat <= Category.SIXES:
                        # Only take if >= 3 of a kind?
                        if score >= cat * 3: return 'score', cat
            
            # If nothing good, dump best points or 0
            for cat in Category.ALL:
                if scorecard.get_score(cat) is None:
                    val = scorecard.calculate_score(cat, dice_values)
                    if val > best_score:
                        best_score = val
                        best_cat = cat
            
            return 'score', best_cat
