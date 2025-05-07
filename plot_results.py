import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import os

# Ensure output directory exists
os.makedirs("plots", exist_ok=True)

# Load and plot RL training data
def plot_rl_data(training_log_path="logs/training_log.csv"):
    df = pd.read_csv(training_log_path)
    
    # Plot 1: Cooperation Rate over Iterations
    plt.figure(figsize=(10, 6))
    plt.plot(df["iteration"], df["coop_rate"], label="Cooperation Rate", color="blue")
    plt.xlabel("Iteration")
    plt.ylabel("Cooperation Rate")
    plt.title("Cooperation Rate During RL Training")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/coop_rate.png")
    plt.close()
    
    # Plot 2: Mean Reward over Iterations
    plt.figure(figsize=(10, 6))
    plt.plot(df["iteration"], df["mean_reward"], label="Mean Reward", color="green")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward During RL Training")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/mean_reward.png")
    plt.close()
    
    # Plot 3: Interactive Cooperation Rate (Plotly)
    fig = px.line(df, x="iteration", y="coop_rate", title="Cooperation Rate During RL Training")
    fig.update_layout(xaxis_title="Iteration", yaxis_title="Cooperation Rate")
    fig.write_to_html("plots/coop_rate_interactive.html")
    
    # Plot 4: Reward Distribution (Histogram)
    plt.figure(figsize=(10, 6))
    plt.hist(df["mean_reward"], bins=20, color="green", alpha=0.7)
    plt.xlabel("Mean Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution Across RL Training")
    plt.grid(True)
    plt.savefig("plots/reward_distribution.png")
    plt.close()

# Plot evaluation comparisons
def plot_evaluation_data(eval_data_path="viz_data/evaluation_data.csv"):
    df = pd.read_csv(eval_data_path)
    
    # Bar plot for cooperation rate
    plt.figure(figsize=(10, 6))
    sns.barplot(x="policy", y="mean_coop_rate", yerr=df["std_coop_rate"], capsize=0.2, palette="Blues")
    plt.xlabel("Policy")
    plt.ylabel("Mean Cooperation Rate")
    plt.title("Cooperation Rate by Policy")
    plt.savefig("plots/eval_coop_rate.png")
    plt.close()
    
    # Bar plot for mean reward
    plt.figure(figsize=(10, 6))
    sns.barplot(x="policy", y="mean_reward", yerr=df["std_reward"], capsize=0.2, palette="Greens")
    plt.xlabel("Policy")
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward by Policy")
    plt.savefig("plots/eval_mean_reward.png")
    plt.close()

# Plot predictor training history
def plot_predictor_data(history_path="logs/predictor_training_history.csv"):
    df = pd.read_csv(history_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df["accuracy"], label="Training Accuracy", color="blue")
    plt.plot(df["val_accuracy"], label="Validation Accuracy", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Predictor Model Training History")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/predictor_training_history.png")
    plt.close()

# Generate replay animation data (simplified for dashboard)
def generate_replay_data(data_path="data/predictor_data.csv"):
    df = pd.read_csv(data_path)
    # Take one episode (first 100 rows, assuming max_rounds=100)
    episode_df = df.iloc[:100][["resource_pool", "action_0", "action_1"]]
    episode_df["round"] = range(1, len(episode_df) + 1)
    episode_df.to_csv("viz_data/replay_data.csv", index=False)
    return episode_df

# Main
if __name__ == "__main__":
    plot_rl_data()
    plot_evaluation_data()
    plot_predictor_data()
    replay_df = generate_replay_data()
    print("Plots and replay data generated in plots/ and viz_data/")