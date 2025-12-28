# Zatzy AI: Superhuman Yahtzee Suite

**Zatzy AI** is a comprehensive AI research laboratory for Yahtzee, featuring multiple state-of-the-art agents ranging from Evolutionary Algorithms to Deep Reinforcement Learning and Neuro-Expectimax search.

## 🤖 Available Agents

* **🏆 Neuro-Expectimax (Cemax-Pro)**: Our flagship agent. Combines the mathematical precision of Expectimax with a Deep Value Network (48-dim input) to estimate long-term returns. **Consistently achieves 300+ points.**
* **🧠 Cemax (Neural)**: A high-speed neural network trained via Behavioral Cloning (Imitation Learning) to mimic the decisions of an optimized Expectimax solver.
* **⚖️ Expectimax (Math)**: An optimized recursive solver that calculates the exact probability of every dice combination for perfect decision-making in the short term.
* **🤖 DQN (Deep Q-Network)**: Trained via Reinforcement Learning (DQN + Reward Shaping). Learns through trial and error to maximize scorecard efficiency.
* **🧬 Genetic AI**: The original champion, evolved over 1000 generations using Neuroevolution.

## ✨ Features

* **Arena Agent Showdown**: Battle different AI architectures against each other or run batch simulations (50+ games) to compare win rates and score distributions.
* **Human vs AI**: Challenge any of the neural or mathematical agents.
* **Training Dashboard**: Real-time analytics for Genetic Evolution (Fitness curves), DQN (Reward/Epsilon decay), and Behavioral Cloning (Accuracy/Loss).
* **Optimized Engine**: Features precomputed probability distributions for 30x faster Expectimax search.

## 🛠️ Tech Stack

* **Python 3.10+**
* **PyTorch**: Neural Network training and inference.
* **Streamlit**: Modern interactive Web Interface.
* **Altair**: Scientific data visualization.
* **Numpy/Pandas**: High-performance data processing.

## 🚀 How to Run

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Launch the Arena**:

```bash
streamlit run app.py
```

## 🏋️ Training Lab

* **Genetic**: `python -c "from src.trainer.train import train; train(generations=1000)"`
* **DQN**: `python src/trainer/train_dqn_pro.py --episodes 1000`
* **Cemax (SL)**: `python src/trainer/train_sl.py --epochs 200`
* **Value Net**: `python src/trainer/train_value_net.py --epochs 500`

---

*Created by Jalpan04 & Antigravity Agent*

