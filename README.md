# Zatzy AI: Genetically Evolved Yahtzee Player

**Zatzy AI** is a Python-based project that uses **Genetic Algorithms (Neuroevolution)** to train a Neural Network to play the game of Yahtzee at a superhuman level. 

Starting from complete randomness, the AI evolved over **1000 Generations** to learn complex strategies like maximizing the Upper Bonus, hunting for Yahtzees, and recognizing Straights—without ever being explicitly told the rules of the game.

## Features

*   **Human vs AI Mode**: Challenge the best evolved agent to a 13-round match.
*   **Watch AI Play**: visualize the AI's decision-making process in real-time.
*   **Interactive Dashboard**: Explore the training history with scientific learning curves (Altair).
*   **God View Engine**: The AI uses a "God View" input vector (48 features) including potential score lookaheads and pattern recognition flags.

## Tech Stack

*   **Python 3.10+**
*   **PyTorch**: Neural Network architecture.
*   **Streamlit**: Interactive Web UI.
*   **Altair**: Data Visualization.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/zatzy-ai.git
    cd zatzy-ai
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Run the Streamlit app:
```bash
streamlit run app.py
```

## Training the AI

If you want to re-train the AI from scratch:
```bash
python -c "from src.trainer.train import train; train(generations=1000, pop_size=200)"
```
*Note: This generates `training_log.json` and saves checkpoints to `checkpoints/`.*

## Deployment (Streamlit Cloud)

This project is ready for **Streamlit Community Cloud**!

1.  Push this code to a **GitHub Repository**.
2.  Go to [share.streamlit.io](https://share.streamlit.io/).
3.  Click **"New App"**.
4.  Select your GitHub Repo.
5.  Set Main File path: `app.py`
6.  Click **Deploy**!

---
*Created by [Your Name] & Antigravity Agent*
