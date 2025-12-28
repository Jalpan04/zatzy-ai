import streamlit as st
import torch
import numpy as np
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.getcwd())

import src.config as config
import src.ai.model as model_module
import src.ai.agent as agent_module
import src.game.engine as engine_module
import src.game.scorecard as scorecard_module
import src.ai.expectimax as expectimax_module
import src.ai.dqn as dqn_module
import src.ui.arena_view as arena_view_module
import importlib

# Force Reload All Modules in Dependency Order
importlib.reload(config)
importlib.reload(model_module)
importlib.reload(agent_module)
importlib.reload(engine_module)
importlib.reload(scorecard_module)
importlib.reload(expectimax_module)
importlib.reload(dqn_module)
importlib.reload(arena_view_module)

from src.ai.model import YahtzeeNetwork
from src.ai.agent import Agent
from src.ai.expectimax import ExpectimaxAgent
from src.ai.dqn import DQNAgent
from src.ui.arena_view import run_arena_view
from src.game.engine import GameEngine
from src.game.scorecard import Category, Scorecard

st.set_page_config(page_title="Zatzy AI", layout="wide")

import src.config as config

st.title("Zatzy AI: Genetic Evolution")

def load_agent(checkpoint_path):
    # Dynamic Loading based on Checkpoint Size
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Infer sizes from state_dict
    # First layer 'network.0.weight' shape is [hidden, input]
    hidden_size = config.HIDDEN_SIZE
    input_size = config.INPUT_SIZE
    
    if 'network.0.weight' in state_dict:
        hidden_size = state_dict['network.0.weight'].shape[0]
        input_size = state_dict['network.0.weight'].shape[1]
        
    model = YahtzeeNetwork(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(state_dict)
    return Agent(model)

def load_dqn_agent():
    # Find best DQN checkpoint
    if not os.path.exists("checkpoints_dqn"):
        return None
    checkpoints = sorted([f for f in os.listdir("checkpoints_dqn") if f.endswith(".pth")], key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if not checkpoints:
        return None
        
    latest_cp = checkpoints[-1]
    cp_path = os.path.join("checkpoints_dqn", latest_cp)
    
    # Load Dict first to check size
    state_dict = torch.load(cp_path, map_location=torch.device('cpu'))
    
    # Infer Hidden Size and Input Size from policy net
    hidden_size = config.HIDDEN_SIZE
    input_size = config.INPUT_SIZE
    is_native_dqn = False
    
    if 'network.0.weight' in state_dict:
        hidden_size = state_dict['network.0.weight'].shape[0]
        input_size = state_dict['network.0.weight'].shape[1]
    elif 'fc1.weight' in state_dict:
        hidden_size = state_dict['fc1.weight'].shape[0]
        input_size = state_dict['fc1.weight'].shape[1]
        is_native_dqn = True
    
    agent = DQNAgent() 
    
    # If the checkpoint size doesn't match the current config, we need to rebuild the nets
    if input_size != config.INPUT_SIZE or hidden_size != agent.policy_net.network[0].out_features if not is_native_dqn else True:
         from src.ai.dqn import DQN
         # Note: Native DQN class in src/ai/dqn.py might need input_size too if it's hardcoded.
         # Assuming YahtzeeNetwork is used for non-native.
         if not is_native_dqn:
            agent.policy_net = YahtzeeNetwork(input_size=input_size, hidden_size=hidden_size)
            agent.target_net = YahtzeeNetwork(input_size=input_size, hidden_size=hidden_size)
    
    try:
        agent.policy_net.load_state_dict(state_dict)
    except RuntimeError:
        # Fallback or detailed error log
        st.error(f"DQN Load Failed: Checkpoint input={input_size}, expected={config.INPUT_SIZE}")
        pass

    agent.epsilon = 0.0 # Force Exploitation
    return agent

def load_sl_agent():
    # Cemax Agent (Neural)
    cp_path = "checkpoints_sl/sl_model_final.pth"
    if not os.path.exists(cp_path):
        return None
    
    from src.trainer.train_sl import YahtzeeNetwork
    # Infer hidden size and input size
    state_dict = torch.load(cp_path, map_location=torch.device('cpu'))
    hidden_size = state_dict['network.0.weight'].shape[0]
    input_size = state_dict['network.0.weight'].shape[1]
    
    model = YahtzeeNetwork(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(state_dict)
    model.eval()
    
    from src.ai.agent import GeneticAgent
    return GeneticAgent(model)

def load_neuro_agent():
    # Superhuman Neuro-Expectimax Agent
    from src.ai.neuro_expectimax import NeuroExpectimaxAgent
    cp_path = "checkpoints_value/value_net_final.pth"
    # Even if CP doesn't exist, the agent handles it (untrained net)
    return NeuroExpectimaxAgent(cp_path)

def render_dice(dice_values):
    cols = st.columns(5)
    for i, val in enumerate(dice_values):
        with cols[i]:
            # Simple SVG or Unicode
            # Unicode Dice: \u2680 (1) to \u2685 (6)
            unicode_map = {0: '?', 1: '\u2680', 2: '\u2681', 3: '\u2682', 4: '\u2683', 5: '\u2684', 6: '\u2685'}
            st.markdown(f"<div style='font-size: 50px; text-align: center;'>{unicode_map.get(val, '?')}</div>", unsafe_allow_html=True)

def render_scorecard(scorecard):
    # Create DataFrame
    data = []
    for cat in Category.ALL:
        name = Category.NAME_MAP[cat]
        score = scorecard.get_score(cat)
        data.append({"Category": name, "Score": str(score) if score is not None else "-"})
    
    st.table(pd.DataFrame(data))
    st.metric("Total Score", scorecard.get_total_score())

# Sidebar
st.sidebar.header("Control Panel")
mode = st.sidebar.radio("Mode", ["Watch AI Play", "Training Dashboard", "Play vs AI", "Arena: Agent Showdown"])

def init_vs_game():
    if "vs_human_engine" not in st.session_state:
        st.session_state.vs_human_engine = GameEngine()
        st.session_state.vs_ai_engine = GameEngine()
        st.session_state.vs_turn = 1
        st.session_state.vs_game_over = False
        # Load best agent automatically
        checkpoints = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pkl")])
        if checkpoints:
            st.session_state.vs_ai_agent = load_agent(os.path.join("checkpoints", checkpoints[-1]))
        else:
             st.error("No AI Agent found!")

if mode == "Arena: Agent Showdown":
    run_arena_view(load_agent, load_dqn_agent, ExpectimaxAgent)

elif mode == "Play vs AI":
    st.subheader("Human vs AI")
    
    with st.expander("How to Play", expanded=True):
        st.markdown("""
        1. **Roll Dice**: You have 3 rolls per turn.
        2. **Keep Dice**: Check the box under the dice you want to SAVE.
        3. **Score**: Click a button to lock in your score for that category.
        4. **Goal**: Get the highest total score after 13 rounds!
        """)

    # AI Selector
    ai_type = st.selectbox("Choose Opponent", ["Genetic AI (Best)", "Expectimax (Math)", "DQN (Reinforcement)", "Cemax (Neural)", "Neuro-Expectimax (300+)"])

    # Reset game if AI type changes
    if "vs_ai_type" not in st.session_state:
        st.session_state.vs_ai_type = ai_type
    
    if st.session_state.vs_ai_type != ai_type:
        st.session_state.vs_ai_type = ai_type
        if "vs_human_engine" in st.session_state:
            del st.session_state.vs_human_engine
            del st.session_state.vs_ai_engine
            # Force reload
            st.rerun()

    # Initialization
    if "vs_human_engine" not in st.session_state:
        st.session_state.vs_human_engine = GameEngine()
        st.session_state.vs_ai_engine = GameEngine()
        st.session_state.vs_logs = []
        
        # Load Agent based on selection
        if ai_type == "Expectimax (Math)":
            st.session_state.vs_ai_agent = ExpectimaxAgent()
        elif ai_type == "Cemax (Neural)":
             sl = load_sl_agent()
             if sl:
                 st.session_state.vs_ai_agent = sl
             else:
                 st.error("No Cemax Model found! Train it first.")
                 st.session_state.vs_ai_agent = ExpectimaxAgent()
        elif "Neuro-Expectimax" in ai_type:
             st.session_state.vs_ai_agent = load_neuro_agent()
        elif ai_type == "DQN (Reinforcement)":
            dqn = load_dqn_agent()
            if dqn:
                st.session_state.vs_ai_agent = dqn
            else:
                 st.error("No DQN Checkpoints found! Train it first.")
                 st.session_state.vs_ai_agent = ExpectimaxAgent() # Fallback
        else:
            checkpoints = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pkl")])
            if checkpoints:
                st.session_state.vs_ai_agent = load_agent(os.path.join("checkpoints", checkpoints[-1]))
            else:
                st.session_state.vs_ai_agent = ExpectimaxAgent() # Fallback

    human = st.session_state.vs_human_engine
    ai = st.session_state.vs_ai_engine
    
    # Restart Button
    if st.button("Restart New Game"):
        del st.session_state.vs_human_engine
        del st.session_state.vs_ai_engine
        st.rerun()

    col1, col2 = st.columns(2)
    
    # --- HUMAN COLUMN ---
    with col1:
        st.markdown(f"### You (Score: {human.scorecard.get_total_score()})")
        st.caption(f"Round {human.turn_number} / 13 | Rolls Left: {human.rolls_left}")
        
        # --- DICE & KEEP UI ---
        # We need a form or persistent state for checkboxes? 
        # Streamlit checkboxes persist if key is stable.
        
        current_dice = human.dice.values
        unicode_map = {0: '?', 1: '\u2680', 2: '\u2681', 3: '\u2682', 4: '\u2683', 5: '\u2684', 6: '\u2685'}
        
        # Container for Dice
        st.markdown("##### Your Dice")
        d_cols = st.columns(5)
        keep_indices = []
        
        for i, val in enumerate(current_dice):
            with d_cols[i]:
                # Big Dice Emoji - REPLACED WITH UNICODE SYMBOL ONLY, NO EMOJIS
                st.markdown(f"<div style='font-size: 50px; line-height: 1; text-align: center;'>{unicode_map.get(val, '?')}</div>", unsafe_allow_html=True)
                
                # Checkbox
                # Only show checkbox if we have rolls left and not game over
                if human.rolls_left > 0 and not human.game_over:
                    # Key must be unique per turn/roll state to reset correctly? 
                    # Actually, if we want them to uncheck after a roll, we change the key.
                    # Or we manually reset? Simpler: Key includes Turn+RollsLeft.
                    if st.checkbox("Hold", key=f"hold_{i}_{human.turn_number}_{human.rolls_left}"):
                        keep_indices.append(i)
        
        st.write("---")
        
        # --- ACTIONS ---
        if not human.game_over:
            # 1. ROLL BUTTON
            if human.rolls_left > 0:
                roll_btn = st.button("Roll Dice", use_container_width=True, type="primary")
                if roll_btn:
                    human.dice.roll(set(keep_indices))
                    human.rolls_left -= 1
                    st.rerun()
            else:
                st.warning("No rolls left! You must score.")
            
            # 2. SCORE BUTTONS
            st.write("#### Choose Category to Score")
            available_cats = [c for c in Category.ALL if human.scorecard.get_score(c) is None]
            
            # Group into columns for compactness
            sc_cols = st.columns(3)
            for idx, cat in enumerate(available_cats):
                cat_name = Category.NAME_MAP[cat]
                potential_score = human.scorecard.calculate_score(cat, human.dice.values)
                
                # Highlight high scores?
                btn_str = f"**{cat_name}**\n\n(+{potential_score})"
                
                with sc_cols[idx % 3]:
                    if st.button(btn_str, key=f"score_{cat}", use_container_width=True):
                        human.apply_action('score', cat)
                        
                        # --- AI TURN (Instant) ---
                        if not ai.game_over:
                            ai_log = []
                            ai_turn = ai.turn_number
                            while ai.turn_number == ai_turn and not ai.game_over:
                                # Safe guard: Ensure AI doesn't loop infinitely if logic fails
                                if ai.rolls_left < 0: break
                                
                                state = ai.get_state_vector()
                                mask = ai.get_mask()
                                # Pass 'engine=ai' for Expectimax lookups
                                action_type, action_val = st.session_state.vs_ai_agent.select_action(state, mask, engine=ai)
                                
                                engine_module.GameEngine.apply_action(ai, action_type, action_val)
                                
                                if action_type == 'score':
                                    cat_n = Category.NAME_MAP[action_val]
                                    pts = ai.scorecard.get_score(action_val)
                                    st.session_state.vs_logs.append(f"Round {ai_turn}: AI scored {pts} pts in {cat_n}")
                                    break
                        
                        st.rerun()
        else:
            st.success("Your Game Finished!")

        render_scorecard(human.scorecard)

    # --- AI COLUMN ---
    with col2:
        st.markdown(f"### AI (Score: {ai.scorecard.get_total_score()})")
        st.caption(f"Round {ai.turn_number} / 13 | AI plays automatically")
        
        render_dice(ai.dice.values)
        
        if st.session_state.vs_logs:
            st.info(f"**Last Action:** {st.session_state.vs_logs[-1]}")
            
        render_scorecard(ai.scorecard)
    
    # Winner Declaration
    if human.game_over and ai.game_over:
        h_score = human.scorecard.get_total_score()
        a_score = ai.scorecard.get_total_score()
        
        st.write("---")
        if h_score > a_score:
            st.balloons()
            st.title(f"YOU WIN! ({h_score} vs {a_score})")
        else:
            st.error(f"AI WINS! ({a_score} vs {h_score})")

elif mode == "Watch AI Play":
    st.subheader("AI Gameplay Viewer")
    
    # Select Agent Type
    agent_type = st.sidebar.radio("Agent Type", ["Genetic AI", "Expectimax (Math)", "DQN (Reinforcement)", "Cemax (Neural)", "Neuro-Expectimax (300+)"])
    
    if agent_type == "Expectimax (Math)" or agent_type == "DQN (Reinforcement)" or "Cemax" in agent_type or "Neuro" in agent_type:
        if st.button("Run Game"):
            if agent_type == "Expectimax (Math)":
                agent = ExpectimaxAgent()
            elif agent_type == "Cemax (Neural)":
                agent = load_sl_agent()
                if not agent:
                    st.error("No Cemax Model found! Run training.")
                    st.stop()
            elif "Neuro" in agent_type:
                agent = load_neuro_agent()
            else:
                agent = load_dqn_agent()
                if not agent:
                    st.error("No DQN Checkpoint found!")
                    st.stop()
            
            engine = GameEngine()
            
            # ... Copy paste loop logic ...
            game_container = st.empty()
            max_steps = 100
            step = 0
            
            while not engine.game_over and step < max_steps:
                 state = engine.get_state_vector()
                 mask = engine.get_mask()
                 action_type, action_val = agent.select_action(state, mask, engine=engine)
                 
                 # Render State
                 with game_container.container():
                     st.write(f"**Turn {engine.turn_number}** | Rolls Left: {engine.rolls_left}")
                     render_dice(engine.dice.values)
                     
                     if action_type == 'keep':
                        keep_str = bin(action_val)[2:].zfill(5)[::-1] 
                        st.info(f"AI Decision: **KEEP** (Mask: {keep_str})")
                     else:
                        cat_n = Category.NAME_MAP[action_val]
                        st.success(f"AI Decision: **SCORE** {cat_n}")
                     
                     render_scorecard(engine.scorecard)
                 
                 time.sleep(1.0 if engine.rolls_left == 3 else 0.2) # Faster for Math bot
                 
                 # Apply
                 engine.apply_action(action_type, action_val)
                 step += 1
             
            st.success(f"Game Over! Final Score: {engine.scorecard.get_total_score()}")
            render_scorecard(engine.scorecard)
    
    else:
        # Checkpoints logic for Genetic
        if not os.path.exists("checkpoints"):
            st.error("No checkpoints found. Run training first.")
        else:
            checkpoints = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pkl")])
            if not checkpoints:
                st.warning("No .pkl checkpoints in checkpoints/")
            else:
                selected_cp = st.sidebar.selectbox("Load Agent", checkpoints, index=len(checkpoints)-1)
                
                if st.button("Run Game"):
                    agent = load_agent(os.path.join("checkpoints", selected_cp))
                    engine = GameEngine()
                    # Visualization Loop - REPEATED LOGIC (Refactor later?)
                    game_container = st.empty()
                    max_steps = 100
                    step = 0
                    
                    while not engine.game_over and step < max_steps:
                        state = engine.get_state_vector()
                        mask = engine.get_mask()
                        action_type, action_val = agent.select_action(state, mask) # Genetic doesn't need engine
                        
                        # Render State
                        with game_container.container():
                            st.write(f"**Turn {engine.turn_number}** | Rolls Left: {engine.rolls_left}")
                            render_dice(engine.dice.values)
                            
                            if action_type == 'keep':
                                keep_str = bin(action_val)[2:].zfill(5)[::-1]
                                st.info(f"AI Decision: **KEEP** (Mask: {keep_str})")
                            else:
                                st.success(f"AI Decision: **SCORE** Category {action_val}")
                            
                            render_scorecard(engine.scorecard)
                        
                        time.sleep(1.0 if engine.rolls_left == 3 else 0.5)
                        
                        # Apply
                        engine.apply_action(action_type, action_val)
                        step += 1
                    
                    st.success(f"Game Over! Final Score: {engine.scorecard.get_total_score()}")
                    render_scorecard(engine.scorecard)

elif mode == "Training Dashboard":
    st.title("AI Brain Analytics")
    st.markdown("Analyze the training performance of our Artificial Intelligence models.")
    
    # Tabs for Different Models
    tab_gen, tab_dqn, tab_sl = st.tabs(["Genetic Algorithm", "Deep Q-Network", "Cemax (SL)"])
    
    import json
    import pandas as pd
    import altair as alt

    # --- GENETIC TAB ---
    with tab_gen:
        if not os.path.exists("training_log.json"):
            st.warning("No Genetic training log found.")
        else:
            with open("training_log.json", "r") as f:
                history_gen = json.load(f)
            df_gen = pd.DataFrame(history_gen)
            
            # Metrics
            cols = st.columns(4)
            best_all_time = df_gen['best'].max()
            current_avg = df_gen['average'].iloc[-1]
            improvement = current_avg - df_gen['average'].iloc[0]
            total_gens = len(df_gen)
            
            cols[0].metric("All-Time Best Score", f"{best_all_time}")
            cols[1].metric("Latest Avg Score", f"{current_avg:.1f}")
            cols[2].metric("Total Improvement", f"{improvement:+.1f}")
            cols[3].metric("Generations", f"{total_gens}")
            
            st.write("### Evolutionary Progress")
            
            # Main Chart: Area (Range) + Line (Best/Avg)
            base = alt.Chart(df_gen).encode(x=alt.X('generation', title='Generation'))
            
            line_avg = base.mark_line(color='#4c78a8', size=3).encode(
                y=alt.Y('average', title='Score'),
                tooltip=['generation', 'average', 'best', 'worst']
            )
            line_best = base.mark_line(color='#f58518', size=2).encode(y='best')
            area_range = base.mark_area(opacity=0.3, color='#72b7b2').encode(y='worst', y2='best')
            
            chart_gen = (area_range + line_avg + line_best).properties(height=400).interactive()
            st.altair_chart(chart_gen, use_container_width=True)
            
            st.caption("Orange: Best | Blue: Average | Green Area: Population Range")

    # --- DQN TAB ---
    with tab_dqn:
        if not os.path.exists("dqn_training_log.json"):
            st.info("No DQN training log found.")
        else:
            with open("dqn_training_log.json", "r") as f:
                history_dqn = json.load(f)
            df_dqn = pd.DataFrame(history_dqn)
            
            # Calculate Rolling Average (Trends)
            df_dqn['rolling_avg'] = df_dqn['score'].rolling(window=20).mean()
            
            # Metrics
            d_cols = st.columns(4)
            dqn_best = df_dqn['score'].max()
            dqn_curr_eps = df_dqn['epsilon'].iloc[-1] if 'epsilon' in df_dqn else 0
            dqn_last_10_avg = df_dqn['score'].tail(10).mean()
            dqn_episodes = len(df_dqn)
            
            d_cols[0].metric("Best Episode Score", f"{dqn_best}")
            d_cols[1].metric("Recent Avg (Last 10)", f"{dqn_last_10_avg:.1f}")
            d_cols[2].metric("Exploration Rate", f"{dqn_curr_eps:.2f}")
            d_cols[3].metric("Total Episodes", f"{dqn_episodes}")
            
            st.write("### Reinforcement Learning Curve")
            
            # Chart 1: Score vs Episode
            base_dqn = alt.Chart(df_dqn).encode(x=alt.X('episode', title='Episode'))
            
            points = base_dqn.mark_circle(opacity=0.3, size=30, color='gray').encode(
                y=alt.Y('score', title='Score'),
                tooltip=['episode', 'score', 'epsilon']
            )
            trend = base_dqn.mark_line(color='#e45756', size=3).encode(
                y=alt.Y('rolling_avg', title='20-Ep Rolling Avg')
            )
            
            chart_dqn = (points + trend).properties(height=400).interactive()
            st.altair_chart(chart_dqn, use_container_width=True)
            
            st.write("### Exploration Decay")
            chart_eps = alt.Chart(df_dqn).mark_line(color='purple').encode(
                x='episode',
                y='epsilon',
                tooltip=['episode', 'epsilon']
            ).properties(height=200).interactive()
            st.altair_chart(chart_eps, use_container_width=True)
            
    with tab_sl:
        if not os.path.exists("checkpoints_sl/training_log.json"):
            st.info("No Cemax logs found. Run training script.")
        else:
            with open("checkpoints_sl/training_log.json", "r") as f:
                 history_sl = json.load(f)
            df_sl = pd.DataFrame(history_sl)
            
            c1, c2 = st.columns(2)
            c1.metric("Final Accuracy", f"{df_sl['accuracy'].iloc[-1]:.2f}%")
            c2.metric("Final Loss", f"{df_sl['loss'].iloc[-1]:.4f}")
            
            st.write("### Accuracy Curve")
            chart_sl = alt.Chart(df_sl).mark_line(point=True).encode(
                x='epoch',
                y=alt.Y('accuracy', scale=alt.Scale(domain=[0, 100])),
                tooltip=['epoch', 'accuracy', 'loss']
            ).interactive()
            st.altair_chart(chart_sl, use_container_width=True)
            
            st.write("### Loss Curve")
            chart_loss = alt.Chart(df_sl).mark_line(color='red').encode(
                x='epoch',
                y='loss',
                 tooltip=['epoch', 'accuracy', 'loss']
            ).interactive()
            st.altair_chart(chart_loss, use_container_width=True)

elif mode == "Arena: Agent Showdown":
    run_arena_view(load_agent, load_dqn_agent, ExpectimaxAgent, load_sl_agent, load_neuro_agent)
