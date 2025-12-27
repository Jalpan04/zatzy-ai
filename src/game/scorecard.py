from typing import List, Dict, Optional
from collections import Counter

class Category:
    ONES = 1
    TWOS = 2
    THREES = 3
    FOURS = 4
    FIVES = 5
    SIXES = 6
    THREE_OF_A_KIND = 7
    FOUR_OF_A_KIND = 8
    FULL_HOUSE = 9
    SMALL_STRAIGHT = 10
    LARGE_STRAIGHT = 11
    YAHTZEE = 12
    CHANCE = 13

    ALL = [ONES, TWOS, THREES, FOURS, FIVES, SIXES, 
           THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE, 
           SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE]
    
    UPPER = [ONES, TWOS, THREES, FOURS, FIVES, SIXES]
    
    NAME_MAP = {
        1: "ONES", 2: "TWOS", 3: "THREES", 4: "FOURS", 5: "FIVES", 6: "SIXES",
        7: "THREE_OF_A_KIND", 8: "FOUR_OF_A_KIND", 9: "FULL_HOUSE",
        10: "SMALL_STRAIGHT", 11: "LARGE_STRAIGHT", 12: "YAHTZEE", 13: "CHANCE"
    }

class Scorecard:
    def __init__(self):
        self.scores: Dict[int, int] = {} # Map Category ID -> Score
        self.yahtzee_bonus = 0
        
    def is_full(self) -> bool:
        return len(self.scores) == 13

    def get_score(self, category: int) -> Optional[int]:
        return self.scores.get(category)
        
    def score(self, category: int, dice_values: List[int]) -> int:
        """
        Records the score for a category. Raises ValueError if already filled.
        Returns the points scored for this turn (excluding bonuses calculated at end, 
        but including immediate Yahtzee bonus if applicable).
        """
        if category in self.scores:
            raise ValueError(f"Category {category} already filled.")
            
        points = self.calculate_score(category, dice_values)
        
        # Handle multiple Yahtzees (Bonus Rule)
        # If we already have a Yahtzee scored (50 points) and we roll another one:
        if self._is_yahtzee(dice_values) and self.scores.get(Category.YAHTZEE) == 50:
            self.yahtzee_bonus += 100
            # Note: The choice of category is still up to the caller. 
            # Standard rules say you must use it in Upper if available, else Joker.
            # Here we just record the score for the chosen category.
            # If the user chose a Lower category and it's a Yahtzee, standard rules usually apply max points.
            # Our calculate_score logic handles the 'Joker' effect for Full House/Straights 
            # if we implement the check effectively.
            
            # Joker Rule Enforcement inside calculation is tricky because it depends on game state.
            # For simplicity in this v1: We assume the AI/Player chooses a valid category.
            # If they pick Full House with a Yahtzee, calculate_score checks dice.
        
        self.scores[category] = points
        return points

    def calculate_score(self, category: int, dice: List[int]) -> int:
        """Calculates potential score for a category given dice."""
        counts = Counter(dice)
        
        # Upper Section
        if category == Category.ONES: return counts[1] * 1
        if category == Category.TWOS: return counts[2] * 2
        if category == Category.THREES: return counts[3] * 3
        if category == Category.FOURS: return counts[4] * 4
        if category == Category.FIVES: return counts[5] * 5
        if category == Category.SIXES: return counts[6] * 6
        
        # Lower Section
        total = sum(dice)
        
        if category == Category.THREE_OF_A_KIND:
            return total if any(c >= 3 for c in counts.values()) else 0
            
        if category == Category.FOUR_OF_A_KIND:
            return total if any(c >= 4 for c in counts.values()) else 0
            
        if category == Category.FULL_HOUSE:
            # Standard: 3 of one, 2 of another. Or 5 of same (Yahtzee counts as Full House in some variations/Joker)
            # Joker rule usually allows Yahtzee to score as Full House (25).
            has_3 = any(c >= 3 for c in counts.values())
            has_2 = any(c >= 2 for c in counts.values())
            # A set of 5 counts as 3 (and 2 is implied if we split it, but explicitly: Counter({5:5}) -> values=[5])
            # strict definition: needs a 3 and a 2.
            # If 5 of a kind: It satisfies Full House conceptually in Joker rules.
            is_yahtzee = any(c == 5 for c in counts.values())
            
            if (has_3 and len(counts) == 2) or is_yahtzee:
                return 25
            return 0
            
        if category == Category.SMALL_STRAIGHT:
            # 4 consecutive dice
            unique = sorted(set(dice))
            # Check for generic sub-sequences 1-2-3-4, 2-3-4-5, 3-4-5-6
            # Simplest: convert to set, check intersection
            s = set(unique)
            if {1,2,3,4}.issubset(s) or {2,3,4,5}.issubset(s) or {3,4,5,6}.issubset(s):
                return 30
            # Joker rule for Yahtzee: If 5 of a kind, does it count? 
            # Standard Rules: Yes, if YAHTZEE box is filled, additional Yahtzees can be used as wildcards for straights.
            if self._is_yahtzee(dice) and self.scores.get(Category.YAHTZEE) == 50:
                 return 30
            return 0
            
        if category == Category.LARGE_STRAIGHT:
            unique = sorted(set(dice))
            s = set(unique)
            if {1,2,3,4,5}.issubset(s) or {2,3,4,5,6}.issubset(s):
                return 40
            if self._is_yahtzee(dice) and self.scores.get(Category.YAHTZEE) == 50:
                 return 40
            return 0
            
        if category == Category.YAHTZEE:
            return 50 if self._is_yahtzee(dice) else 0
            
        if category == Category.CHANCE:
            return total
            
        return 0

    def get_total_score(self) -> int:
        total = sum(self.scores.values()) + self.yahtzee_bonus
        
        # Upper Bonus
        upper_score = 0
        for cat in Category.UPPER:
            upper_score += self.scores.get(cat, 0)
        
        if upper_score >= 63:
            total += 35
            
        return total

    def _is_yahtzee(self, dice: List[int]) -> bool:
        return any(c == 5 for c in Counter(dice).values())
