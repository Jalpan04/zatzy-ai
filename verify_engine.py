import sys
import os
sys.path.append(os.getcwd())
from src.game.engine import GameEngine

engine = GameEngine()
state = engine.get_state_vector()
print(f"State Vector Size: {len(state)}")
print(f"State Vector: {state}")
