# Global Game & AI Configuration

# AI Model Hyperparameters
INPUT_SIZE = 61    # Adds 13 Actual Score Fields
HIDDEN_SIZE = 512  # Higher capacity for complex regression
OUTPUT_SIZE = 45   # 32 (Keep Masks) + 13 (Score Categories)

# Training Hyperparameters
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.15
ELITISM_PCT = 0.1
