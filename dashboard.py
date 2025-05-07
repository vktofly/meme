import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="Multi-Agent RL Dashboard", layout="wide")

# Title and description
st.title("Multi-Agent Reinforcement Learning Dashboard")
st.markdown("""
This dashboard visualizes the results of a multi-agent reinforcement learning project for a resource-sharing game.
It includes cooperation trends, policy evaluations, and neural network prediction performance.
""")

# Load data
@st.cache_data
def load_data():
    rl_data = pd.read_csv("viz_data/rl_training_data.csv")
    eval_data = pd.read_csv("viz_data/evaluation_data.csv")
    predictor_history = pd.read_csv("viz_data/predictor_history.csv")
    predictor_metrics = pd.read_csv("viz_data/predictor_metrics.csv")
    replay_data = pd.read_csv("viz_data/replay_data.csv")
    return rl_data, eval_data, predictor_history, predictor_metrics, replay_data

rl_data, eval_data, predictor_history, predictor_metrics, replay_data = load_data()

# Sidebar for controls
st.sidebar.header("Controls")
iteration_range = st.sidebar.slider("Select Iteration Range", 0, int(rl_data["iteration"].max()), (0, int(rl_data["iteration"].max())))
policy_select = st.sidebar.multiselect("Select Policies to Compare", eval_data["policy"].tolist(), default=eval_data["policy"].tolist())
replay_round = st.sidebar.slider("Select Round for Replay", 1, len(replay_data), 1)

# Section 1: RL Training Trends
st.header("RL Training Trends")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cooperation Rate")
    fig = px.line(rl_data[(rl_data["iteration"] >= iteration_range[0]) & (rl_data["iteration"] <= iteration_range[1])],
                  x="iteration", y="coop_rate_smoothed", title="Smoothed Cooperation Rate")
    fig.update_layout(xaxis_title="Iteration", yaxis_title="Cooperation Rate")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Mean Reward")
    fig = px.line(rl_data[(rl_data["iteration"] >= iteration_range[0]) & (rl_data["iteration"] <= iteration_range[1])],
                  x="iteration", y="mean_reward_smoothed", title="Smoothed Mean Reward")
    fig.update_layout(xaxis_title="Iteration", yaxis_title="Mean Reward")
    st.plotly_chart(fig, use_container_width=True)

# Section 2: Policy Evaluation
st.header("Policy Evaluation")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cooperation Rate by Policy")
    filtered_eval_data = eval_data[eval_data["policy"].isin(policy_select)]
    fig = go.Figure(data=[
        go.Bar(x=filtered_eval_data["policy"], y=filtered_eval_data["mean_coop_rate"],
               error_y=dict(type="data", array=filtered_eval_data["std_coop_rate"]))
    ])
    fig.update_layout(title="Mean Cooperation Rate", xaxis_title="Policy", yaxis_title="Cooperation Rate")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Mean Reward by Policy")
    fig = go.Figure(data=[
        go.Bar(x=filtered_eval_data["policy"], y=filtered_eval_data["mean_reward"],
               error_y=dict(type="data", array=filtered_eval_data["std_reward"]))
    ])
    fig.update_layout(title="Mean Reward", xaxis_title="Policy", yaxis_title="Mean Reward")
    st.plotly_chart(fig, use_container_width=True)

# Section 3: Predictor Performance
st.header("Neural Network Predictor Performance")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Training History")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predictor_history.index, y=predictor_history["accuracy"], mode="lines", name="Training Accuracy"))
    fig.add_trace(go.Scatter(x=predictor_history.index, y=predictor_history["val_accuracy"], mode="lines", name="Validation Accuracy"))
    fig.update_layout(title="Predictor Training History", xaxis_title="Epoch", yaxis_title="Accuracy")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Confusion Matrix")
    if os.path.exists("plots/predictor_confusion_matrix.png"):
        img = Image.open("plots/predictor_confusion_matrix.png")
        st.image(img, caption="Confusion Matrix for Action Prediction", use_column_width=True)
    else:
        st.write("Confusion matrix not found.")

# Section 4: Replay Animation
st.header("Episode Replay")
st.subheader(f"Round {replay_round}")
current_round = replay_data[replay_data["round"] == replay_round]
if not current_round.empty:
    st.write(f"Resource Pool: {current_round['resource_pool'].iloc[0]:.2f}")
    st.write(f"Agent 0 Action: {current_round['action_0'].iloc[0]}")
    st.write(f"Agent 1 Action: {current_round['action_1'].iloc[0]}")
    fig = px.line(replay_data[replay_data["round"] <= replay_round], x="round", y="resource_pool", title="Resource Pool Over Rounds")
    fig.update_layout(xaxis_title="Round", yaxis_title="Resource Pool")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No data for selected round.")