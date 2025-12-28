import random
from src.game.scorecard import Category

class RandomAgent:
    def __init__(self):
        self.name = "Random (Baseline)"
        
    def select_action(self, state, mask, engine):
        # 1. Enumerate Legal Actions
        legal_actions = []
        if engine.rolls_left > 0:
            for i in range(32):
                if mask[i]: legal_actions.append(('keep', i))
        else:
            for i in range(13):
                if mask[32+i]: legal_actions.append(('score', i+1))
        
        return random.choice(legal_actions)

class GreedyAgent:
    def __init__(self):
        self.name = "Greedy (Naive)"
        
    def select_action(self, state, mask, engine):
        """
        Greedy Strategy:
        - If Keep phase: Random Keep (Naive).
        - If Score phase: Pick Max Points using current dice.
        """
        if engine.rolls_left > 0:
            # Naive Keep: Just keep everything (Stop rolling) or random?
            # A true "Greedy" in Yahtzee usually tries to maximize expected score of THAT category.
            # But the simplest "Naive" baseline is: Reroll everything unless we have 5 of a kind?
            # Let's do: Keep nothing (Full Reroll) until forced to score.
            # Actually, "Greedy" usually refers to scoring decision.
            # Let's implement: "Maximize Immediate Score".
            # For Keeping: It's hard to be greedy without lookahead.
            # So we will use Random Keep, Greedy Score.
            
            # Better baseline: "Full Reroll" (Keep nothing)
            if mask[0]: return ('keep', 0)
            return ('keep', random.choice([i for i in range(32) if mask[i]]))
            
        else:
            # Score Phase: Pick highest point value available
            best_score = -1
            best_cat = -1
            
            for i in range(13):
                if mask[32+i]:
                    cat = i + 1
                    s = engine.scorecard.calculate_score(cat, engine.dice.values)
                    if s > best_score:
                        best_score = s
                        best_cat = cat
            
            return ('score', best_cat)
