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
import importlib

# Force Reload All Modules in Dependency Order
importlib.reload(config)
importlib.reload(model_module)
importlib.reload(agent_module)
importlib.reload(engine_module)
importlib.reload(scorecard_module)

from src.ai.model import YahtzeeNetwork
from src.ai.agent import Agent
from src.game.engine import GameEngine
from src.game.scorecard import Category, Scorecard

st.set_page_config(page_title="Zatzy AI", layout="wide")

import src.config as config

st.title("🧬 Zatzy AI: Genetic Evolution")

def load_agent(checkpoint_path):
    # Use Global Config for robust dimensions
    model = YahtzeeNetwork(input_size=config.INPUT_SIZE, hidden_size=config.HIDDEN_SIZE)
    # Handle if CPU/CUDA
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return Agent(model)

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
        data.append({"Category": name, "Score": score if score is not None else "-"})
    
    st.table(pd.DataFrame(data))
    st.metric("Total Score", scorecard.get_total_score())

# Sidebar
st.sidebar.header("Control Panel")
mode = st.sidebar.radio("Mode", ["Watch AI Play", "Training Dashboard", "Play vs AI"])

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

if mode == "Play vs AI":
    st.subheader("⚔️ Human vs AI")
    
    with st.expander("ℹ️ How to Play", expanded=True):
        st.markdown("""
        1. **Roll Dice**: You have 3 rolls per turn.
        2. **Keep Dice**: Check the box under the dice you want to SAVE.
        3. **Score**: Click a button to lock in your score for that category.
        4. **Goal**: Get the highest total score after 13 rounds!
        """)

    # Initialization
    if "vs_human_engine" not in st.session_state:
        st.session_state.vs_human_engine = GameEngine()
        st.session_state.vs_ai_engine = GameEngine()
        st.session_state.vs_logs = []
        
        # Load best agent
        checkpoints = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pkl")])
        if checkpoints:
            st.session_state.vs_ai_agent = load_agent(os.path.join("checkpoints", checkpoints[-1]))

    human = st.session_state.vs_human_engine
    ai = st.session_state.vs_ai_engine
    
    # Restart Button
    if st.button("🔄 Restart New Game"):
        del st.session_state.vs_human_engine
        del st.session_state.vs_ai_engine
        st.rerun()

    col1, col2 = st.columns(2)
    
    # --- HUMAN COLUMN ---
    with col1:
        st.markdown(f"### 👤 You (Score: {human.scorecard.get_total_score()})")
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
                # Big Dice Emoji
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
                roll_btn = st.button("🎲 Roll Dice", use_container_width=True, type="primary")
                if roll_btn:
                    human.dice.roll(set(keep_indices))
                    human.rolls_left -= 1
                    st.rerun()
            else:
                st.warning("⚠️ No rolls left! You must score.")
            
            # 2. SCORE BUTTONS
            st.write("#### 👇 Choose Category to Score")
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
                                action_type, action_val = st.session_state.vs_ai_agent.select_action(state, mask)
                                
                                engine_module.GameEngine.apply_action(ai, action_type, action_val)
                                
                                if action_type == 'score':
                                    cat_n = Category.NAME_MAP[action_val]
                                    pts = ai.scorecard.get_score(action_val)
                                    st.session_state.vs_logs.append(f"Round {ai_turn}: AI scored {pts} pts in {cat_n}")
                                    break
                        
                        st.rerun()
        else:
            st.success("✅ Your Game Finished!")

        render_scorecard(human.scorecard)

    # --- AI COLUMN ---
    with col2:
        st.markdown(f"### 🤖 AI (Score: {ai.scorecard.get_total_score()})")
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
            st.title(f"🏆 YOU WIN! ({h_score} vs {a_score})")
        else:
            st.error(f"💀 AI WINS! ({a_score} vs {h_score})")

elif mode == "Watch AI Play":
    st.subheader("🤖 AI Gameplay Viewer")
    
    # Checkpoints
    if not os.path.exists("checkpoints"):
        st.error("No checkpoints found. Run training first.")
    else:
        checkpoints = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pkl")])
        if not checkpoints:
             st.warning("No .pkl checkpoints in checkpoints/")
        else:
             selected_cp = st.sidebar.selectbox("Load Agent", checkpoints, index=len(checkpoints)-1)
             
             if st.button("▶️ Run Game"):
                 agent = load_agent(os.path.join("checkpoints", selected_cp))
                 engine = GameEngine()
                 
                 # Visualization Loop
                 game_container = st.empty()
                 
                 max_steps = 100
                 step = 0
                 
                 # We need to capture the game flow.
                 # Since Streamlit reruns script on interaction, 'animation' is done via time.sleep inside a loop
                 # and updating a single container.
                 
                 logs = []
                 
                 while not engine.game_over and step < max_steps:
                     state = engine.get_state_vector()
                     mask = engine.get_mask()
                     action_type, action_val = agent.select_action(state, mask)
                     
                     # Render before action? Or after?
                     # Render State
                     with game_container.container():
                         st.write(f"**Turn {engine.turn_number}** | Rolls Left: {engine.rolls_left}")
                         render_dice(engine.dice.values)
                         
                         if action_type == 'keep':
                            keep_str = bin(action_val)[2:].zfill(5)[::-1] # Binary string (visual debug)
                            st.info(f"AI Decision: **KEEP** (Mask: {keep_str})")
                         else:
                            st.success(f"AI Decision: **SCORE** Category {action_val}")
                         
                         render_scorecard(engine.scorecard)
                     
                     time.sleep(1.0 if engine.rolls_left == 3 else 0.5) # Slow down for viewing
                     
                     # Apply
                     engine.apply_action(action_type, action_val)
                     step += 1
                 
                 st.success(f"Game Over! Final Score: {engine.scorecard.get_total_score()}")
                 render_scorecard(engine.scorecard)

elif mode == "Training Dashboard":
    st.subheader("📈 Training Progress (Interactive)")
    
    import json
    import pandas as pd
    import altair as alt
    
    if not os.path.exists("training_log.json"):
        st.warning("No training log found. Run training first to generate stats.")
    else:
        with open("training_log.json", "r") as f:
            history = json.load(f)
            
        df = pd.DataFrame(history)
        
        # KPIS
        col1, col2, col3 = st.columns(3)
        col1.metric("Highest AI Score", f"{df['best'].max()}", delta=f"{df['best'].max() - df['best'].iloc[0]:.0f}")
        col2.metric("Current Avg Score", f"{df['average'].iloc[-1]}", delta=f"{df['average'].iloc[-1] - df['average'].iloc[-0]:.0f}")
        col3.metric("Generations Trained", f"{len(df)}")
        
        st.write("---")
        
        # MELT dataframe for Altair (Long Format)
        df_melted = df.melt('generation', var_name='Metric', value_name='Score')
        # Filter to keep only interesting lines
        df_melted = df_melted[df_melted['Metric'].isin(['best', 'average', 'worst'])]

        # Altair Chart
        st.write("### 🧠 AI Learning Curve")
        
        base = alt.Chart(df).encode(x=alt.X('generation', title='Generation'))

        # Line for Average
        line_avg = base.mark_line(color='#1f77b4', size=3).encode(
            y=alt.Y('average', title='Score'),
            tooltip=['generation', 'average', 'best']
        )
        
        # Line for Best
        line_best = base.mark_line(color='#ff7f0e', size=2).encode(
            y='best'
        )
        
        # Area for Range (Worst to Best)
        band = base.mark_area(opacity=0.2, color='#1f77b4').encode(
            y='worst',
            y2='best'
        )
        
        chart = (band + line_avg + line_best).properties(
            height=400
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
        
        st.caption("Orange: Best Performer | Blue: Population Average | Shaded: Full Range")

        # 3. Stats Table
        with st.expander("🔍 View Raw Training Data"):
            st.dataframe(df)
