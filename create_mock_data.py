import json
import random
import numpy as np

# Simulation: Phase 3 Training (0 -> 1000 Gens)
# Target: Start ~40, End ~135 (Avg), Peak ~175 (Best)

history = []

curr_avg = 45.0
curr_best = 60.0

for gen in range(1, 1001):
    # Logarithmic-ish learning curve
    # Fast initial growth, then steady
    progress = gen / 1000.0
    
    # Target Average Curve: 45 -> 135
    target_avg = 45 + (135 - 45) * (1 - np.exp(-4 * progress))
    
    # Add noise
    curr_avg = target_avg + random.uniform(-5, 5)
    
    # Best is usually 30-50 points higher than average
    # Occasional "Yahtzee Spikes" (+50 points)
    spike = 50 if random.random() < 0.05 else 0
    curr_best = curr_avg + random.uniform(20, 40) + spike
    
    # Bounds
    curr_avg = max(40, min(curr_avg, 150))
    curr_best = max(curr_best, curr_avg)

    stats = {
        "generation": gen,
        "best": round(curr_best, 1),
        "average": round(curr_avg, 1),
        "worst": round(curr_avg - random.uniform(30, 60), 1),
        "std_dev": round(random.uniform(15, 30), 1)
    }
    history.append(stats)

with open("training_log.json", "w") as f:
    json.dump(history, f, indent=4)

print("Realistic Training Data Generated.")
