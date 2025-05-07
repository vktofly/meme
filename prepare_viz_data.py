import pandas as pd
import numpy as np
import os

# Ensure output directory exists
os.makedirs("viz_data", exist_ok=True)

# Load and preprocess RL training data
def process_rl_data(training_log_path="logs/training_log.csv"):
    df = pd.read_csv(training_log_path)
    # Smooth cooperation rate and reward for better visualization
    df["coop_rate_smoothed"] = df["coop_rate"].rolling(window=5, min_periods=1).mean()
    df["mean_reward_smoothed"] = df["mean_reward"].rolling(window=5, min_periods=1).mean()
    df.to_csv("viz_data/rl_training_data.csv", index=False)
    return df

# Load and preprocess evaluation data
def process_evaluation_data(eval_results_path="logs/evaluation_results.txt"):
    eval_data = {
        "policy": [],
        "mean_coop_rate": [],
        "std_coop_rate": [],
        "mean_reward": [],
        "std_reward": [],
        "mean_prediction_accuracy": []
    }
    with open(eval_results_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "Trained Policy" in line:
                eval_data["policy"].append("Trained")
                eval_data["mean_coop_rate"].append(float(lines[i+1].split(": ")[1].split(" ± ")[0]))
                eval_data["std_coop_rate"].append(float(lines[i+1].split(" ± ")[1]))
                eval_data["mean_reward"].append(float(lines[i+2].split(": ")[1].split(" ± ")[0]))
                eval_data["std_reward"].append(float(lines[i+2].split(" ± ")[1]))
                eval_data["mean_prediction_accuracy"].append(None)
            elif "Random Baseline" in line:
                eval_data["policy"].append("Random")
                eval_data["mean_coop_rate"].append(float(lines[i+1].split(": ")[1].split(" ± ")[0]))
                eval_data["std_coop_rate"].append(float(lines[i+1].split(" ± ")[1]))
                eval_data["mean_reward"].append(float(lines[i+2].split(": ")[1].split(" ± ")[0]))
                eval_data["std_reward"].append(float(lines[i+2].split(" ± ")[1]))
                eval_data["mean_prediction_accuracy"].append(None)
            elif "Greedy Baseline" in line:
                eval_data["policy"].append("Greedy")
                eval_data["mean_coop_rate"].append(float(lines[i+1].split(": ")[1].split(" ± ")[0]))
                eval_data["std_coop_rate"].append(float(lines[i+1].split(" ± ")[1]))
                eval_data["mean_reward"].append(float(lines[i+2].split(": ")[1].split(" ± ")[0]))
                eval_data["std_reward"].append(float(lines[i+2].split(" ± ")[1]))
                eval_data["mean_prediction_accuracy"].append(None)
            elif "Predictive Agent" in line:
                eval_data["policy"].append("Predictive")
                eval_data["mean_coop_rate"].append(float(lines[i+1].split(": ")[1].split(" ± ")[0]))
                eval_data["std_coop_rate"].append(float(lines[i+1].split(" ± ")[1]))
                eval_data["mean_reward"].append(float(lines[i+2].split(": ")[1].split(" ± ")[0]))
                eval_data["std_reward"].append(float(lines[i+2].split(" ± ")[1]))
                eval_data["mean_prediction_accuracy"].append(float(lines[i+3].split(": ")[1]))
    
    df = pd.DataFrame(eval_data)
    df.to_csv("viz_data/evaluation_data.csv", index=False)
    return df

# Load and preprocess predictor data
def process_predictor_data(history_path="logs/predictor_training_history.csv", metrics_path="logs/predictor_metrics.txt"):
    # Training history
    df_history = pd.read_csv(history_path)
    df_history.to_csv("viz_data/predictor_history.csv", index=False)
    
    # Metrics
    metrics = {}
    with open(metrics_path, "r") as f:
        for line in f:
            key, value = line.strip().split(": ")
            metrics[key] = float(value)
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("viz_data/predictor_metrics.csv", index=False)
    return df_history, metrics_df

# Main
if __name__ == "__main__":
    rl_df = process_rl_data()
    eval_df = process_evaluation_data()
    predictor_history_df, predictor_metrics_df = process_predictor_data()
    print("Visualization data prepared and saved in viz_data/")