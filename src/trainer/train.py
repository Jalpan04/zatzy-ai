import torch
import numpy as np
import os
import pickle
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.ai.model import YahtzeeNetwork
from src.ai.agent import Agent
from src.ai.genetics import params_to_vector, vector_to_params, crossover, mutate
from src.trainer.evaluator import Evaluator

def tournament_select(fitnesses, k=3):
    """
    Selects k individuals at random and returns the index of the best one.
    fitnesses is a list of (original_index, score).
    """
    candidates_indices = np.random.randint(0, len(fitnesses), k)
    candidates = [fitnesses[i] for i in candidates_indices]
    winner = max(candidates, key=lambda x: x[1])
    return winner[0]

def train(generations=100, pop_size=50, games_per_eval=5):
    print(f"Starting Training: Gens={generations}, Pop={pop_size}, Games/Eval={games_per_eval}")
    
    # Initialize Population
    population = [YahtzeeNetwork() for _ in range(pop_size)]
    evaluator = Evaluator()
    
    os.makedirs("checkpoints", exist_ok=True)
    
    # Use Config for Mutation
    from src import config
    
    import json
    
    training_history = []
    
    for gen in range(1, generations + 1):
        fitnesses = []
        
        # Evaluate
        scores = []
        for i, model in enumerate(population):
            agent = Agent(model)
            score = evaluator.evaluate(agent, num_games=games_per_eval)
            fitnesses.append((i, score))
            scores.append(score)
        
        # Statistics
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        std_dev = np.std(scores)
        
        # Log Stats
        stats = {
            "generation": gen,
            "best": float(max_score),
            "average": float(avg_score),
            "worst": float(min_score),
            "std_dev": float(std_dev)
        }
        training_history.append(stats)
        
        # Save Log to File (Overwrite each time for safety)
        with open("training_log.json", "w") as f:
            json.dump(training_history, f, indent=4)
        
        # Sort by fitness (Descending)
        fitnesses.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Gen {gen}: Best={max_score:.1f}, Avg={avg_score:.1f}, Min={min_score:.1f}")
        
        # Elitism: Keep Top 10%
        num_elites = max(2, int(pop_size * 0.1))
        # ... (Rest of elitism code) ...
        elites_indices = [idx for idx, score in fitnesses[:num_elites]]
        elites = []
        for idx in elites_indices:
            # Clone to ensure they persist unchanged
            new_model = YahtzeeNetwork()
            new_model.load_state_dict(population[idx].state_dict())
            elites.append(new_model)
            
        # Selection and Next Gen
        new_population = list(elites)
        
        while len(new_population) < pop_size:
            # Tournament Selection
            p1_idx = tournament_select(fitnesses)
            p2_idx = tournament_select(fitnesses)
            
            p1_vec = params_to_vector(population[p1_idx])
            p2_vec = params_to_vector(population[p2_idx])
            
            # Crossover
            child_vec = crossover(p1_vec, p2_vec)
            
            # Mutation
            child_vec = mutate(child_vec, mutation_rate=config.MUTATION_RATE, mutation_strength=config.MUTATION_STRENGTH)
            
            child_model = YahtzeeNetwork()
            vector_to_params(child_model, child_vec)
            new_population.append(child_model)
            
        population = new_population
        
        # Save Checkpoint
        if gen % 10 == 0 or gen == generations:
             # Save the BEST model of this generation
             # 'elites[0]' IS the best model from the previous generation
             
             save_path = f"checkpoints/gen_{gen}_score_{int(max_score)}.pkl"
             torch.save(elites[0].state_dict(), save_path)
             print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train()
