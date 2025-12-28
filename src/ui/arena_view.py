import streamlit as st
import pandas as pd
import altair as alt
import time
from src.game.engine import GameEngine
# Imports for Agents will be passed or imported here

def run_arena_view(load_agent_func, load_dqn_agent_func, ExpectimaxAgent):
    st.subheader("⚔️ AI Battle Arena")
    
    col1, col2 = st.columns(2)
    with col1:
        agent1_name = st.selectbox("Player 1", ["Expectimax (Math)", "Genetic AI (Best)", "DQN (Reinforcement)"], index=0)
    with col2:
        agent2_name = st.selectbox("Player 2", ["Expectimax (Math)", "Genetic AI (Best)", "DQN (Reinforcement)"], index=2)
        
    n_games = st.slider("Number of Games", 10, 100, 50)
    
    if st.button("Start Battle"):
        # Load Agents
        a1 = load_agent_by_name(agent1_name, load_agent_func, load_dqn_agent_func, ExpectimaxAgent)
        a2 = load_agent_by_name(agent2_name, load_agent_func, load_dqn_agent_func, ExpectimaxAgent)
        
        if not a1 or not a2:
            st.error("Could not load one of the agents. Check checkpoints.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        # Run Simulation
        for i in range(n_games):
            # Agent 1 Game
            e1 = GameEngine()
            run_fast_game(e1, a1)
            s1 = e1.scorecard.get_total_score()
            
            # Agent 2 Game
            e2 = GameEngine()
            run_fast_game(e2, a2)
            s2 = e2.scorecard.get_total_score()
            
            results.append({"Game": i+1, "Agent": agent1_name, "Score": s1, "Winner": agent1_name if s1 > s2 else agent2_name})
            results.append({"Game": i+1, "Agent": agent2_name, "Score": s2, "Winner": agent1_name if s1 > s2 else agent2_name})
            
            progress_bar.progress((i + 1) / n_games)
            status_text.text(f"Simulating Game {i+1}/{n_games}...")
            
        st.success("Battle Complete!")
        
        # Analysis
        df = pd.DataFrame(results)
        
        # 1. Score Comparison
        st.write("### Score Distribution")
        chart = alt.Chart(df).mark_boxplot().encode(
            x='Agent',
            y='Score',
            color='Agent'
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        
        # 2. Win Rate
        st.write("### Head-to-Head Win Rate")
        wins = df[df['Score'] == df.groupby("Game")['Score'].transform('max')] # Tie handling is simple
        # Actually easier:
        win_counts = df.groupby(["Game", "Winner"]).size().reset_index().groupby("Winner").size().reset_index(name='Wins')
        
        win_chart = alt.Chart(win_counts).mark_bar().encode(
            x='Winner',
            y='Wins',
            color='Winner'
        )
        st.altair_chart(win_chart, use_container_width=True)
        
        # Stats Table
        st.write("### Statistics")
        stats = df.groupby("Agent")['Score'].agg(['mean', 'max', 'min', 'std']).reset_index()
        st.dataframe(stats)

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
