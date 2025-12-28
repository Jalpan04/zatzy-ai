
import sys, os
sys.path.append(os.getcwd())

import pandas as pd
import time
import numpy as np
import torch
from src.game.engine import GameEngine

# Import Agents
from src.ai.baselines import RandomAgent, GreedyAgent
from src.ai.rule_based import RuleBasedAgent
from src.ai.mcts import MCTSAgent
from src.ai.dqn import DQNAgent, DQN
from src.ai.expectimax import ExpectimaxAgent
from src.ai.neuro_expectimax import NeuroExpectimaxAgent
import src.config as config

def load_dqn():
    import src.config as config
    # Legacy DQN trained on 48 features
    input_size = 48 
    try:
        agent = DQNAgent(input_size=input_size, output_size=config.OUTPUT_SIZE)
        path = "checkpoints_dqn/dqn_pro_best_252.pth"
        if os.path.exists(path):
            state = torch.load(path, map_location='cpu')
            agent.policy_net.load_state_dict(state)
            agent.target_net.load_state_dict(state)
            agent.epsilon = 0.0
            return agent
    except Exception as e:
         print(f"DQN Init/Load Failed: {e}")
    return None

def load_genetic():
    from src.ai.agent import Agent
    from src.ai.model import YahtzeeNetwork
    import src.config as config
    # Genetic also likely 48 features
    input_size = 48
    try:
        model = YahtzeeNetwork(input_size=input_size, hidden_size=config.HIDDEN_SIZE, output_size=config.OUTPUT_SIZE)
        path = "checkpoints/gen_1000_score_170.pkl"
        if os.path.exists(path):
            data = torch.load(path, map_location='cpu')
            if isinstance(data, torch.Tensor):
                from src.ai.genetics import vector_to_params
                vector_to_params(model, data)
            elif isinstance(data, dict):
                model.load_state_dict(data)
            return Agent(model)
    except Exception as e:
        print(f"Genetic Load Error: {e}")
    return None

def run_tournament(num_games=100, output_file="research_results.csv"):
    agents = []
    
    # 1. Baselines
    agents.append(RandomAgent())
    agents.append(GreedyAgent())
    
    # 2. Rule Based
    agents.append(RuleBasedAgent())
    
    # 3. Simulation
    agents.append(MCTSAgent(simulations=5)) # Reduced sims 20->5 for speed
    
    # 4. Learning (DQN)
    dqn = load_dqn()
    if dqn: 
        dqn.name = "DQN (Reinforcement Learning)"
        agents.append(dqn)
        
    # 5. Learning (Genetic)
    gen = load_genetic()
    if gen:
        gen.name = "Genetic (Evolutionary)"
        agents.append(gen)
    
    # 6. Heuristic (Champion)
    agents.append(ExpectimaxAgent())
    
    results = []
    
    print(f"🏆 Starting Grand Tournament ({num_games} games per agent)...")
    
    for agent in agents:
        print(f"Running {agent.name}...")
        start_time_agent = time.time()
        
        for i in range(num_games):
            engine = GameEngine()
            game_start = time.time()
            
            while not engine.game_over:
                state = engine.get_state_vector()
                mask = engine.get_mask()
                
                # Handle different select_action signatures?
                # Most accept (state, mask, engine)
                try:
                    action_type, action_val = agent.select_action(state, mask, engine)
                    engine.apply_action(action_type, action_val)
                except Exception as e:
                    print(f"Error agent {agent.name}: {e}")
                    break
            
            game_end = time.time()
            score = engine.scorecard.get_total_score()
            
            # Record Metrics
            results.append({
                "Agent": agent.name,
                "GameID": i,
                "Score": score,
                "Time": game_end - game_start,
                "Yahtzee_Count": engine.scorecard.scores.get(12, 0) == 50, # Cat 12 is Yahtzee
                "Bonus_Achieved": engine.scorecard.get_total_score() >= (sum(engine.scorecard.scores.values()) + 35)
                # Check bonus properly: Logic is inside get_total_score but hard to extract boolean.
                # Approximate:
            })
            
            print(f"\r  Game {i+1}/{num_games} | Score: {score}", end="")
            
        print(f"\n  ✅ Average: {np.mean([r['Score'] for r in results if r['Agent'] == agent.name]):.2f}")
        
    # Save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\n📄 Results saved to {output_file}")

if __name__ == "__main__":
    run_tournament()
