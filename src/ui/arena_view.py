import streamlit as st
import pandas as pd
import altair as alt
import time
from src.game.engine import GameEngine
# Imports for Agents will be passed or imported here

    st.subheader("⚔️ AI Battle Arena")
    
    col1, col2 = st.columns(2)
    with col1:
        agent1_name = st.selectbox("Player 1 (Left)", ["Expectimax (Math)", "Genetic AI (Best)", "DQN (Reinforcement)"], index=0)
    with col2:
        agent2_name = st.selectbox("Player 2 (Right)", ["Expectimax (Math)", "Genetic AI (Best)", "DQN (Reinforcement)"], index=2)

    battle_mode = st.radio("Battle Mode", ["Visual Face-Off (Watch 1 Match)", "Batch Simulation (Stats Only)"])
    
    if battle_mode == "Batch Simulation (Stats Only)":
        n_games = st.slider("Number of Games", 10, 100, 50)
        
        if st.button("Start Batch Simulation"):
            # Load Agents
            a1 = load_agent_by_name(agent1_name, load_agent_func, load_dqn_agent_func, ExpectimaxAgent)
            a2 = load_agent_by_name(agent2_name, load_agent_func, load_dqn_agent_func, ExpectimaxAgent)
            
            if not a1 or not a2: return

            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            for i in range(n_games):
                # Agent 1
                e1 = GameEngine()
                run_fast_game(e1, a1)
                s1 = e1.scorecard.get_total_score()
                
                # Agent 2
                e2 = GameEngine()
                run_fast_game(e2, a2)
                s2 = e2.scorecard.get_total_score()
                
                winner = agent1_name if s1 > s2 else (agent2_name if s2 > s1 else "Tie")
                results.append({"Game": i+1, "P1 Score": s1, "P2 Score": s2, "Winner": winner})
                
                progress_bar.progress((i + 1) / n_games)
            
            st.success("Simulation Complete!")
            df = pd.DataFrame(results)
            
            # Comparison metrics
            avg1 = df['P1 Score'].mean()
            avg2 = df['P2 Score'].mean()
            wins = df['Winner'].value_counts()
            
            m1, m2, m3 = st.columns(3)
            m1.metric(f"{agent1_name} Win Rate", f"{(wins.get(agent1_name, 0)/n_games)*100:.1f}%")
            m2.metric(f"{agent2_name} Win Rate", f"{(wins.get(agent2_name, 0)/n_games)*100:.1f}%")
            m3.metric("Avg Score Diff", f"{avg1 - avg2:.1f}")
            
            st.write("### Score Distribution")
            chart_data = pd.melt(df, id_vars=['Game'], value_vars=['P1 Score', 'P2 Score'], var_name='Agent', value_name='Score')
            chart = alt.Chart(chart_data).mark_boxplot().encode(x='Agent', y='Score', color='Agent')
            st.altair_chart(chart, use_container_width=True)

    else:
        # VISUAL FACE-OFF
        if st.button("Start LIVE Match"):
            a1 = load_agent_by_name(agent1_name, load_agent_func, load_dqn_agent_func, ExpectimaxAgent)
            a2 = load_agent_by_name(agent2_name, load_agent_func, load_dqn_agent_func, ExpectimaxAgent)
            
            if not a1 or not a2: return
            
            e1 = GameEngine()
            e2 = GameEngine()
            
            # Layout
            c1, c2 = st.columns(2)
            c1.markdown(f"### {agent1_name}")
            c2.markdown(f"### {agent2_name}")
            
            container1 = c1.empty()
            container2 = c2.empty()
            
            step = 0
            speed = 0.3
            
            while not (e1.game_over and e2.game_over) and step < 200:
                # Agent 1 Move
                if not e1.game_over:
                    play_one_step(e1, a1)
                    
                # Agent 2 Move
                if not e2.game_over:
                    play_one_step(e2, a2)
                
                # Render
                with container1.container():
                     render_game_state(e1)
                with container2.container():
                     render_game_state(e2)
                     
                time.sleep(speed)
                step += 1
            
            # Final Result
            s1 = e1.scorecard.get_total_score()
            s2 = e2.scorecard.get_total_score()
            
            if s1 > s2:
                c1.success(f"WINNER: {s1}")
                c2.error(f"LOSER: {s2}")
            elif s2 > s1:
                c1.error(f"LOSER: {s1}")
                c2.success(f"WINNER: {s2}")
            else:
                st.info(f"DRAW: {s1}")

def play_one_step(engine, agent):
    # Logic for one step (Roll or Score)
    state = engine.get_state_vector()
    mask = engine.get_mask()
    try:
         action_type, action_val = agent.select_action(state, mask, engine=engine)
    except TypeError:
         action_type, action_val = agent.select_action(state, mask)
    
    engine.apply_action(action_type, action_val)

def render_game_state(engine):
    st.write(f"**Score: {engine.scorecard.get_total_score()}** | Turn: {engine.turn_number}")
    
    # Render Dice (Simple Text/Emoji)
    vals = engine.dice.values
    unicode_map = {0: '?', 1: '\u2680', 2: '\u2681', 3: '\u2682', 4: '\u2683', 5: '\u2684', 6: '\u2685'}
    dice_str = " ".join([unicode_map.get(v, '?') for v in vals])
    st.markdown(f"<div style='font-size: 40px;'>{dice_str}</div>", unsafe_allow_html=True)
    
    # Scorecard Mini
    # Show only filled categories or top 5? Or total?
    # Just show last action logic if possible? No, shows full scorecard briefly?
    # Keeping it simple to fit on screen
    st.progress(min(engine.turn_number / 13.0, 1.0))
    st.caption(f"Rolls: {engine.rolls_left}")

def load_agent_by_name(name, load_genetic, load_dqn, ExpectimaxCls):
    import os # Lazy import
    if name == "Expectimax (Math)":
        return ExpectimaxCls()
    elif name == "DQN (Reinforcement)":
        return load_dqn()
    else:
        # Genetic
        checkpoints = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pkl")])
        if checkpoints:
            return load_genetic(os.path.join("checkpoints", checkpoints[-1]))
    return None

def run_fast_game(engine, agent):
    steps = 0
    while not engine.game_over and steps < 100:
        state = engine.get_state_vector()
        mask = engine.get_mask()
        
        # Handle difference in signatures
        # Genetic: select_action(state, mask)
        # Expectimax/DQN: select_action(state, mask, engine=engine)
        try:
             action_type, action_val = agent.select_action(state, mask, engine=engine)
        except TypeError:
             # Fallback for Genetic which might not accept engine kwarg yet?
             # Actually Genetic Agent wrapper in app.py logic handles this? 
             # No, Genetic Agent class in src/ai/agent.py takes (state, mask).
             action_type, action_val = agent.select_action(state, mask)

        engine.apply_action(action_type, action_val)
        steps += 1
