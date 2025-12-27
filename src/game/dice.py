import random
from typing import List, Set

class Dice:
    def __init__(self):
        self.count = 5
        self.values = [0] * self.count
        self.reset()

    def reset(self):
        """Resets the dice for a new turn (all zeros or immediately rolled).
        Usually, a turn starts with a roll, so we can initialize to random or zeros.
        Using 0 to indicate 'not rolled yet' or just rolling immediately.
        """
        self.values = [0] * self.count

    def roll(self, keep_indices: Set[int] = None):
        """
        Rolls the dice.
        :param keep_indices: A set of indices (0-4) to keep. All others are re-rolled.
        """
        if keep_indices is None:
            keep_indices = set()

        for i in range(self.count):
            if i not in keep_indices:
                self.values[i] = random.randint(1, 6)
    
    def get_values(self) -> List[int]:
        """Returns the current values of the dice."""
        return list(self.values)

    def get_sorted_values(self) -> List[int]:
        """Returns a sorted copy of the dice values (useful for scoring/AI state)."""
        return sorted(self.values)

    def __str__(self):
        return str(self.values)
