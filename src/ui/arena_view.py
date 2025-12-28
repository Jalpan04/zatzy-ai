import streamlit as st
import pandas as pd
import altair as alt
import time
from src.game.engine import GameEngine
# Imports for Agents will be passed or imported here

def run_arena_view(load_agent_func, load_dqn_agent_func, ExpectimaxAgent):
    st.subheader("AI Battle Arena")
    
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
            
            # History Log (Turn-based)
            match_log = {} # {turn_num: {"Agent 1": "...", "Agent 2": "..."}}
            
            # Containers for Last Action (Visual feedback still shows Keeps)
            last_action_c1 = c1.empty()
            last_action_c2 = c2.empty()
            
            while not (e1.game_over and e2.game_over) and step < 400:
                # Agent 1
                a1_desc = ""
                if not e1.game_over:
                    # Capturing turn before move to know where to log? 
                    # Actually apply_action increments turn.
                    # So if we score, the move belongs to 'turn_before_move'.
                    current_turn_1 = e1.turn_number
                    action_type, action_val = play_one_step(e1, a1)
                    
                    if action_type == 'keep':
                        mask_str = bin(action_val)[2:].zfill(5)[::-1]
                        a1_desc = f"Kept: {mask_str}"
                    else:
                        from src.game.scorecard import Category
                        cat_n = Category.NAME_MAP.get(action_val, str(action_val))
                        pts = e1.scorecard.get_score(action_val)
                        a1_desc = f"Scored: {cat_n} ({pts})"
                        # Log to History
                        if current_turn_1 not in match_log: match_log[current_turn_1] = {}
                        match_log[current_turn_1]["Turn"] = current_turn_1
                        match_log[current_turn_1]["Player 1"] = f"{cat_n} ({pts})"
                        match_log[current_turn_1]["p1_pts"] = pts
                
                # Agent 2
                a2_desc = ""
                if not e2.game_over:
                    current_turn_2 = e2.turn_number
                    action_type, action_val = play_one_step(e2, a2)
                    
                    if action_type == 'keep':
                        mask_str = bin(action_val)[2:].zfill(5)[::-1]
                        a2_desc = f"Kept: {mask_str}"
                    else:
                        from src.game.scorecard import Category
                        cat_n = Category.NAME_MAP.get(action_val, str(action_val))
                        pts = e2.scorecard.get_score(action_val)
                        a2_desc = f"Scored: {cat_n} ({pts})"
                        # Log to History
                        if current_turn_2 not in match_log: match_log[current_turn_2] = {}
                        match_log[current_turn_2]["Turn"] = current_turn_2
                        match_log[current_turn_2]["Player 2"] = f"{cat_n} ({pts})"
                        match_log[current_turn_2]["p2_pts"] = pts
                
                # Render Visuals
                with container1.container():
                     render_game_state(e1)
                     if a1_desc: last_action_c1.info(a1_desc)
                with container2.container():
                     render_game_state(e2)
                     if a2_desc: last_action_c2.info(a2_desc)
                     
                time.sleep(speed)
                step += 1
            
            # Final Result
            s1 = e1.scorecard.get_total_score()
            s2 = e2.scorecard.get_total_score()
            
            st.markdown("---")
            st.markdown(f"### WINNER: {agent1_name if s1 > s2 else agent2_name}")
            
            if s1 > s2:
                c1.success(f"WINNER ({s1})")
                c2.error(f"LOSER ({s2})")
            elif s2 > s1:
                c1.error(f"LOSER ({s1})")
                c2.success(f"WINNER ({s2})")
            else:
                st.info(f"DRAW ({s1})")
            
            # Build DataFrame from Match Log
            # Sort by Turn
            sorted_turns = sorted(match_log.keys())
            final_data = []
            
            p1_total_manual = 0
            p2_total_manual = 0
            
            for t in sorted_turns:
                row = match_log[t]
                p1_txt = row.get("Player 1", "-")
                p2_txt = row.get("Player 2", "-")
                
                # Extract points roughly or use stored
                p1_pts = row.get("p1_pts", 0)
                p2_pts = row.get("p2_pts", 0)
                
                p1_total_manual += p1_pts
                p2_total_manual += p2_pts
                
                # Determine Round Winner
                r_win = "-"
                if p1_pts > p2_pts: r_win = "Player 1 🟢"
                elif p2_pts > p1_pts: r_win = "Player 2 🔵"
                
                final_data.append({
                    "Turn": t, 
                    "Player 1": p1_txt, 
                    "Player 2": p2_txt,
                    "Round Winner": r_win
                })
            
            # Add TOTAL Row
            final_winner = "DRAW"
            if s1 > s2: final_winner = "PLAYER 1 🏆"
            elif s2 > s1: final_winner = "PLAYER 2 🏆"
            
            final_data.append({
                "Turn": "TOTAL",
                "Player 1": f"**{s1}**",
                "Player 2": f"**{s2}**",
                "Round Winner": final_winner
            })

            # Show History after match
            st.write("### Match History")
            # Highlight winner in history? No, straightforward table is fine.
            st.dataframe(pd.DataFrame(final_data), use_container_width=True)

def play_one_step(engine, agent):
    # Logic for one step (Roll or Score)
    state = engine.get_state_vector()
    mask = engine.get_mask()
    try:
         action_type, action_val = agent.select_action(state, mask, engine=engine)
    except TypeError:
         action_type, action_val = agent.select_action(state, mask)
    
    engine.apply_action(action_type, action_val)
    return action_type, action_val

def render_game_state(engine):
    st.write(f"**Score: {engine.scorecard.get_total_score()}** | Turn: {engine.turn_number}")
    
    # Render Dice (Simple Text/Emoji)
    vals = engine.dice.values
    unicode_map = {0: '?', 1: '\u2680', 2: '\u2681', 3: '\u2682', 4: '\u2683', 5: '\u2684', 6: '\u2685'}
    dice_str = " ".join([unicode_map.get(v, '?') for v in vals])
    st.markdown(f"<div style='font-size: 40px;'>{dice_str}</div>", unsafe_allow_html=True)
    
    # Scorecard Mini
    st.progress(min(engine.turn_number / 13.0, 1.0))
    st.caption(f"Rolls Left: {engine.rolls_left}")

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
