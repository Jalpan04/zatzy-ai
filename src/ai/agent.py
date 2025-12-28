import torch
import numpy as np

class Agent:
    def __init__(self, model):
        self.model = model
        self.model.eval() # Always eval mode, no dropout/batchnorm training logic needed

    def select_action(self, state, mask, **kwargs):
        """
        Selects the best valid action using the model.
        """
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32)
            logits = self.model(x)
            
            # Apply Mask
            # mask_vector is 1.0 for valid, 0.0 for invalid
            mask = torch.tensor(mask_vector, dtype=torch.float32)
            
            # Set invalid actions to a very large negative number
            # We use -1e9 instead of -inf to avoid NaN issues if everything is masked (shouldnt happen)
            masked_logits = logits + (mask - 1.0) * 1e9
            
            action_idx = torch.argmax(masked_logits).item()
            
            # Decode Action
            if action_idx < 32:
                return 'keep', action_idx
            else:
                # Score actions are 32 to 44 mapping to Categories 1 to 13
                # 32 -> Cat 1
                # 44 -> Cat 13
                return 'score', action_idx - 32 + 1
