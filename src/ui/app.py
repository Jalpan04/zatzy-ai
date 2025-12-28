import streamlit as st
import torch
import numpy as np
import os
import sys
import time
import pandas as pd
import altair as alt
import importlib

# Add project root to path
sys.path.append(os.getcwd())

import src.config as config
import src.ai.model as model_module
import src.ai.agent as agent_module
import src.game.engine as engine_module
import src.game.scorecard as scorecard_module

# Force Reload
importlib.reload(config)
importlib.reload(model_module)
importlib.reload(agent_module)
importlib.reload(engine_module)
importlib.reload(scorecard_module)

from src.ai.model import YahtzeeNetwork
from src.ai.agent import Agent
from src.ai.dqn import DQNAgent
from src.ai.expectimax import ExpectimaxAgent
from src.ai.mcts import MCTSAgent
from src.ai.rule_based import RuleBasedAgent
from src.ai.baselines import RandomAgent
from src.game.engine import GameEngine
from src.game.scorecard import Category, Scorecard

st.set_page_config(page_title="Zatzy AI", layout="wide")

st.title("Zatzy AI: Genetic Evolution")

# --- LOADERS ---
def load_agent(checkpoint_path):
    try:
        model = YahtzeeNetwork(input_size=config.INPUT_SIZE, hidden_size=config.HIDDEN_SIZE)
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        return Agent(model)
    except RuntimeError:
        try:
            model = YahtzeeNetwork(input_size=48, hidden_size=128)
            state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            agent = Agent(model)
            agent.input_size = 48
            return agent
        except RuntimeError:
            model = YahtzeeNetwork(input_size=48, hidden_size=256)
            state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            agent = Agent(model)
            agent.input_size = 48
            return agent

def load_dqn_agent():
    if not os.path.exists("checkpoints_dqn"): return None
    checkpoints = sorted([f for f in os.listdir("checkpoints_dqn") if f.endswith(".pth")], key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if not checkpoints: return None
    
    cp_path = os.path.join("checkpoints_dqn", checkpoints[-1])
    try:
        agent = DQNAgent()
        agent.policy_net.load_state_dict(torch.load(cp_path, map_location=torch.device('cpu')))
    except RuntimeError:
        agent = DQNAgent(input_size=48)
        agent.policy_net.load_state_dict(torch.load(cp_path, map_location=torch.device('cpu')))
        
    agent.epsilon = 0.0
    return agent

# --- UI HELPERS ---
def render_dice(dice_values):
    cols = st.columns(5)
    unicode_map = {0: '?', 1: '\u2680', 2: '\u2681', 3: '\u2682', 4: '\u2683', 5: '\u2684', 6: '\u2685'}
    for i, val in enumerate(dice_values):
        with cols[i]:
            st.markdown(f"<div style='font-size: 40px; text-align: center;'>{unicode_map.get(val, '?')}</div>", unsafe_allow_html=True)

def render_compact_scorecard(scorecard):
    st.write(f"**Score: {scorecard.get_total_score()}**")
    # Show only filled categories logic could go here if needed

# --- NAVIGATION ---
st.sidebar.header("Control Panel")
mode = st.sidebar.radio("Mode", ["Battle Royale", "Play vs AI", "Training Dashboard", "Watch AI Play"])

# --- BATTLE ROYALE ---
if mode == "Battle Royale":
    st.header("Battle Royale: All-Stars Tournament")
    st.write("Simultaneous simulation of all 6 agent architectures.")
    
    if st.button("Start Battle Royale"):
        # 1. Initialize All Agents
        agents = {}
        agents["Genetic AI"] = None
        if os.path.exists("checkpoints"):
            cps = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pkl")])
            if cps: agents["Genetic AI"] = load_agent(os.path.join("checkpoints", cps[-1]))
        
        agents["Expectimax"] = ExpectimaxAgent()
        agents["DQN"] = load_dqn_agent()
        agents["Rule-Based"] = RuleBasedAgent()
        agents["MCTS"] = MCTSAgent(simulations=5) # Low sims for speed
        agents["Random"] = RandomAgent()
        
        # Filter out None agents
        active_agents = {k: v for k, v in agents.items() if v is not None}
        
        # 2. Setup Engines
        engines = {name: GameEngine() for name in active_agents}
        
        # 3. UI Layout
        # Create a grid of stats
        st.write("---")
        cols = st.columns(3)
        metrics = {}
        for idx, (name, _) in enumerate(active_agents.items()):
            with cols[idx % 3]:
                st.subheader(name)
                metrics[name] = st.empty()
                metrics[name].write("Score: 0 | Round: 1")
        
        # 4. Game Loop
        progress = st.progress(0)
        round_num = 1
        max_rounds = 13
        
        while round_num <= max_rounds:
            # Play one full round for everyone
            progress.progress(round_num / 13.0)
            
            # Since we want visual updates, we can do turn-by-turn or round-by-round
            # Let's do round-by-round for speed in a 6-agent sim
            
            for name, agent in active_agents.items():
                eng = engines[name]
                # Play the full turn for this agent
                while eng.turn_number == round_num and not eng.game_over:
                    state = eng.get_state_vector()
                    mask = eng.get_mask()
                    # Uniform call signature: all agents either require or accept engine kwarg
                    act_t, act_v = agent.select_action(state, mask, engine=eng)
                        
                    eng.apply_action(act_t, act_v)
            
            # Update UI after every agent finishes the round
            for name, eng in engines.items():
                metrics[name].write(f"Score: {eng.scorecard.get_total_score()} | Round: {round_num}")
                
            time.sleep(0.5) 
            round_num += 1
            
        # 5. Final Results
        st.success("Tournament Complete!")
        
        results = []
        for name, eng in engines.items():
            results.append({"Agent": name, "Final Score": eng.scorecard.get_total_score()})
            
        df_res = pd.DataFrame(results).sort_values(by="Final Score", ascending=False)
        st.table(df_res)
        
        winner = df_res.iloc[0]
        st.balloons()
        st.header(f"🏆 Winner: {winner['Agent']} ({winner['Final Score']})")


# --- PLAY VS AI ---
elif mode == "Play vs AI":
    st.header("Human vs AI Arena")
    
    # AI Selection
    ai_options = ["Genetic AI", "Expectimax", "DQN", "Rule-Based", "MCTS", "Random"]
    ai_choice = st.selectbox("Select Opponent", ai_options)
    
    # Initialize Game State
    if "pva_human" not in st.session_state:
        st.session_state.pva_human = GameEngine()
        st.session_state.pva_ai = GameEngine()
        st.session_state.pva_agent = None
    
    # Restart Logic
    if st.button("New Game"):
        st.session_state.pva_human = GameEngine()
        st.session_state.pva_ai = GameEngine()
        
        # Load Selected Agent
        if ai_choice == "Genetic AI":
             cps = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pkl")])
             if cps: st.session_state.pva_agent = load_agent(os.path.join("checkpoints", cps[-1]))
             else: st.session_state.pva_agent = RandomAgent()
        elif ai_choice == "Expectimax": st.session_state.pva_agent = ExpectimaxAgent()
        elif ai_choice == "DQN": 
            agent = load_dqn_agent()
            st.session_state.pva_agent = agent if agent else RandomAgent()
        elif ai_choice == "Rule-Based": st.session_state.pva_agent = RuleBasedAgent()
        elif ai_choice == "MCTS": st.session_state.pva_agent = MCTSAgent(simulations=15)
        elif ai_choice == "Random": st.session_state.pva_agent = RandomAgent()
        
        st.rerun()

    human = st.session_state.pva_human
    ai = st.session_state.pva_ai
    agent = st.session_state.pva_agent
    
    # Default agent fallback
    if agent is None:
        agent = RandomAgent()
        st.session_state.pva_agent = agent

    c1, c2 = st.columns(2)
    
    # Human View
    with c1:
        st.subheader("You")
        st.write(f"Score: {human.scorecard.get_total_score()}")
        st.write(f"Round: {human.turn_number} | Rolls: {human.rolls_left}")
        
        # Dice
        d_cols = st.columns(5)
        keep = []
        unicode_map = {0: '?', 1: '\u2680', 2: '\u2681', 3: '\u2682', 4: '\u2683', 5: '\u2684', 6: '\u2685'}
        for i, val in enumerate(human.dice.values):
            with d_cols[i]:
                st.markdown(f"<div style='font-size: 40px;'>{unicode_map[val]}</div>", unsafe_allow_html=True)
                if human.rolls_left > 0 and not human.game_over:
                     if st.checkbox("Hold", key=f"h_{human.turn_number}_{human.rolls_left}_{i}"):
                         keep.append(i)
        
        # Controls
        if not human.game_over:
            if human.rolls_left > 0:
                if st.button("Roll"):
                    human.dice.roll(set(keep))
                    human.rolls_left -= 1
                    st.rerun()
            
            # Score
            open_cats = [c for c in Category.ALL if human.scorecard.get_score(c) is None]
            st.write("Score:")
            cols_s = st.columns(3)
            for idx, cat in enumerate(open_cats):
                with cols_s[idx%3]:
                     pot_score = human.scorecard.calculate_score(cat, human.dice.values)
                     if st.button(f"{Category.NAME_MAP[cat]} (+{pot_score})", key=f"s_{cat}"):
                         human.apply_action('score', cat)
                         
                         # AI Turn (Instant)
                         if not ai.game_over:
                             curr_round = ai.turn_number
                             while ai.turn_number == curr_round and not ai.game_over:
                                 if ai.rolls_left < 0: break
                                 stt = ai.get_state_vector()
                                 msk = ai.get_mask()
                                 # Uniform call signature for safety
                                 at, av = agent.select_action(stt, msk, engine=ai)
                                 
                                 ai.apply_action(at, av)
                         st.rerun()
        else:
            st.info("Game Over")

    # AI View
    with c2:
        st.subheader(f"VS {ai_choice}")
        st.write(f"Score: {ai.scorecard.get_total_score()}")
        render_dice(ai.dice.values)
        st.table(pd.DataFrame([
            {"Cat": Category.NAME_MAP[c], "Val": ai.scorecard.get_score(c) if ai.scorecard.get_score(c) is not None else "-"}
            for c in Category.ALL
        ]))


# --- TRAINING DASHBOARD (IMPROVED) ---
elif mode == "Training Dashboard":
    st.header("Advanced Training Analytics")
    
    import json
    
    # Load Data
    gen_data = []
    dqn_data = []
    
    if os.path.exists("data/training_log.json"):
        with open("data/training_log.json") as f: gen_data = json.load(f)
    if os.path.exists("data/dqn_training_log.json"):
        with open("data/dqn_training_log.json") as f: dqn_data = json.load(f)
        
    df_gen = pd.DataFrame(gen_data)
    df_dqn = pd.DataFrame(dqn_data)
    
    tabs = st.tabs(["Genetic AI", "Deep Q-Network", "Comparative Analysis"])
    
    with tabs[0]:
        st.subheader("Genetic Algorithm Progress")
        if not df_gen.empty:
            st.write(f"Total Generations: {len(df_gen)}")
            st.write(f"Peak Score: {df_gen['best'].max()}")
            
            # Interactive Line Chart with Point Selection
            base = alt.Chart(df_gen).encode(x=alt.X('generation', title='Generation'))
            
            line_avg = base.mark_line(color='#4c78a8').encode(
                y=alt.Y('average', title='Score'),
                tooltip=['generation', 'average', 'best']
            )
            line_best = base.mark_line(color='orange').encode(y='best')
            
            # Add interactive selection
            selection = alt.selection_interval(bind='scales')
            chart = (line_avg + line_best).add_params(selection).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
            st.caption("Blue: Average Score | Orange: Best Score (Scroll to Zoom, Drag to Pan)")
            
    with tabs[1]:
        st.subheader("DQN Learning Curve")
        if not df_dqn.empty:
            df_dqn['MA_20'] = df_dqn['score'].rolling(20).mean()
            
            base = alt.Chart(df_dqn).encode(x=alt.X('episode', title='Episode'))
            
            scatter = base.mark_circle(opacity=0.3, size=30).encode(
                y=alt.Y('score', title='Score'),
                tooltip=['episode', 'score']
            )
            
            trend = base.mark_line(color='red', size=3).encode(
                y=alt.Y('MA_20', title='Moving Average (20)'),
                 tooltip=['episode', 'MA_20']
            )
            
            selection = alt.selection_interval(bind='scales')
            chart = (scatter + trend).add_params(selection).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
            st.caption("Dots: Raw Episode Scores | Red Line: 20-Episode Moving Average")
            
    with tabs[2]:
        st.subheader("Model Comparison")
        if not df_gen.empty and not df_dqn.empty:
            # 1. Bar Chart Comparison
            gen_best = df_gen['best'].max()
            dqn_best = df_dqn['score'].max()
            
            eval_data = pd.DataFrame({
                'Model': ['Genetic AI', 'DQN'],
                'Best Score': [gen_best, dqn_best],
                'Color': ['orange', 'red']
            })
            
            bar_chart = alt.Chart(eval_data).mark_bar().encode(
                x='Model',
                y='Best Score',
                color=alt.Color('Color', scale=None),
                tooltip=['Model', 'Best Score']
            ).properties(height=300, title="Peak Performance Benchmark")
            
            st.altair_chart(bar_chart, use_container_width=True)
            
            # 2. Stability Metrics
            st.write("### Stability Analysis (Lower StdDev = More Stable)")
            gen_std = df_gen['average'].tail(50).std()
            dqn_std = df_dqn['score'].tail(50).std()
            
            col1, col2 = st.columns(2)
            col1.metric("Genetic Stability", f"{gen_std:.2f}")
            col2.metric("DQN Stability", f"{dqn_std:.2f}")
            
# --- WATCH AI PLAY ---
elif mode == "Watch AI Play":
    st.header("Spectator Mode")
    
    agent_name = st.selectbox("Select Agent", [
         "Genetic AI", "Expectimax", "DQN", "Rule-Based", "MCTS", "Random"
    ])
    
    if st.button("Run Simulation"):
        # Instantiate
        if agent_name == "Genetic AI":
             cps = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pkl")])
             if cps: agent = load_agent(os.path.join("checkpoints", cps[-1]))
             else: agent = RandomAgent()
        elif agent_name == "Expectimax": agent = ExpectimaxAgent()
        elif agent_name == "DQN": 
            a = load_dqn_agent()
            agent = a if a else RandomAgent()
        elif agent_name == "Rule-Based": agent = RuleBasedAgent()
        elif agent_name == "MCTS": agent = MCTSAgent(simulations=10)
        elif agent_name == "Random": agent = RandomAgent()
        
        eng = GameEngine()
        ph = st.empty()
        
        while not eng.game_over:
            state = eng.get_state_vector()
            mask = eng.get_mask()
            
            if agent_name == "Genetic AI" or agent_name == "DQN":
                 # These support **kwargs or engine=None, so passing engine is safe
                 at, av = agent.select_action(state, mask, engine=eng)
            else:
                 # Expectimax, Rule-Based, MCTS, Random ALL need engine
                 at, av = agent.select_action(state, mask, engine=eng)
                
            eng.apply_action(at, av)
            
            with ph.container():
                st.write(f"Round: {eng.turn_number}")
                render_dice(eng.dice.values)
                st.write(f"Action: {at} {av}")
                st.write(f"Score: {eng.scorecard.get_total_score()}")
                
            time.sleep(0.2)
        
        st.success(f"Final Score: {eng.scorecard.get_total_score()}")
