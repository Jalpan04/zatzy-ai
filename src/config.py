# Global Game & AI Configuration

# AI Model Hyperparameters
INPUT_SIZE = 48    # 5 (Dice) + 6 (Hist) + 13 (Flags) + 13 (Potentials) + 8 (Patterns) + 3 (Context)
HIDDEN_SIZE = 128  # Capacity of the Neural Network
OUTPUT_SIZE = 45   # 32 (Keep Masks) + 13 (Score Categories)

# Training Hyperparameters
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.15
ELITISM_PCT = 0.1
