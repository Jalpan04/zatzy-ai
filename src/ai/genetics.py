import torch
import numpy as np
import copy

def params_to_vector(model: torch.nn.Module) -> torch.Tensor:
    """Flattens all parameters of a model into a single 1D vector."""
    vec = []
    for param in model.parameters():
        vec.append(param.view(-1))
    return torch.cat(vec)

def vector_to_params(model: torch.nn.Module, vector: torch.Tensor):
    """Loads a 1D vector of parameters into the model."""
    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param.data = vector[pointer:pointer + num_param].view_as(param).data
        pointer += num_param

def get_random_agent(model_class, *args, **kwargs):
    model = model_class(*args, **kwargs)
    # Weights are already random initialized by PyTorch
    return model

def crossover(parent1_vec: torch.Tensor, parent2_vec: torch.Tensor, rate=0.7) -> torch.Tensor:
    """
    Performs uniform crossover.
    :param rate: Probability of taking gene from parent 1. (Usually 0.5 for uniform). 
    Wait, standard uniform is 50/50. 
    Standard Single Point is different.
    Let's do Uniform Parameter Crossover: mask.
    """
    if np.random.random() > 0.9: # 10% chance of no crossover (clone one parent)
        return parent1_vec.clone()

    mask = torch.rand_like(parent1_vec) < 0.5
    child = torch.where(mask, parent1_vec, parent2_vec)
    return child

def mutate(vector: torch.Tensor, mutation_rate=0.05, mutation_strength=0.1) -> torch.Tensor:
    """
    Adds Gaussian noise to parameters.
    :param mutation_rate: Fraction of weights to mutate.
    :param mutation_strength: Standard deviation of noise.
    """
    # Create mask for mutation
    mask = torch.rand_like(vector) < mutation_rate
    noise = torch.randn_like(vector) * mutation_strength
    
    # Apply noise only where mask is true
    mutated_vector = vector + (mask.float() * noise)
    return mutated_vector
