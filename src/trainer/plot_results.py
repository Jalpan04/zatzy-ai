
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_research_results(csv_file="research_results.csv"):
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found!")
        return
        
    df = pd.read_csv(csv_file)
    print(df.groupby('Agent')['Score'].describe())
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Box Plot (Overall Performance Distribution)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Agent', y='Score', data=df, palette="viridis")
    plt.title("Agent Performance Comparison (N=50 Games)")
    plt.ylabel("Final Score")
    plt.axhline(y=300, color='r', linestyle='--', label="300 Point Target")
    plt.legend()
    plt.savefig("agent_comparison_boxplot.png")
    print("Saved agent_comparison_boxplot.png")
    
    # 2. Histogram (Distribution Shape)
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=df, x="Score", hue="Agent", fill=True, common_norm=False, palette="viridis", alpha=0.3)
    plt.title("Score Distribution Density")
    plt.axvline(x=300, color='r', linestyle='--')
    plt.savefig("score_distribution_kde.png")
    print("Saved score_distribution_kde.png")

    # 3. Average Score Bar Chart (The Ranking)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Score", y="Agent", data=df, palette="viridis", errorbar=None)
    plt.title("Average Performance Ranking")
    plt.xlabel("Average Score")
    plt.axvline(x=200, color='r', linestyle='--', label="Pro Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("average_score_ranking.png")
    print("Saved average_score_ranking.png")

    # 4. Bonus Achievement Rate (The "Secret Sauce")
    # Bonus_Achieved is boolean, so mean plays nicely as percentage
    plt.figure(figsize=(10, 6))
    if 'Bonus_Achieved' in df.columns:
        # Ensure it's numeric for aggregation if it's not already
        df['Bonus_Numeric'] = df['Bonus_Achieved'].astype(int)
        sns.barplot(x="Bonus_Numeric", y="Agent", data=df, palette="magma", errorbar=None)
        plt.title("Upper Section Bonus Success Rate")
        plt.xlabel("Success Rate (0.0 - 1.0)")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig("bonus_success_rate.png")
        print("Saved bonus_success_rate.png")

if __name__ == "__main__":
    plot_research_results()
