import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
import numpy as np
import tensorflow as tf
from resource_sharing_env import ResourceSharingEnv
import os

# Register custom environment
def env_creator(env_config):
    return ResourceSharingEnv(**env_config)

register_env("ResourceSharingEnv", env_creator)

# Baseline strategies
class RandomAgent:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(6)
    
    def compute_action(self, obs):
        return self.action_space.sample()

class GreedyAgent:
    def compute_action(self, obs):
        return 5

# Predictive agent using neural network
class PredictiveAgent:
    def __init__(self, model_path="models/predictor_model", scaler_mean_path="models/scaler_mean.npy", scaler_scale_path="models/scaler_scale.npy"):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler_mean = np.load(scaler_mean_path)
        self.scaler_scale = np.load(scaler_scale_path)
    
    def compute_action(self, obs):
        # Extract features
        resource_pool = obs["resource_pool"][0]
        prev_actions = obs["prev_actions"]
        features = np.array([[resource_pool, prev_actions[0], prev_actions[1]]])
        # Scale features
        features_scaled = (features - self.scaler_mean) / self.scaler_scale
        # Predict
        pred_probs = self.model.predict(features_scaled, verbose=0)
        return np.argmax(pred_probs, axis=1)[0]

# Evaluation function
def evaluate_policy(trainer, env, baseline=None, predictive=False, num_episodes=10):
    results = {"coop_rate": [], "total_reward": [], "prediction_accuracy": []}
    for _ in range(num_episodes):
        obs = env.reset()
        done = [False] * env.num_agents
        total_reward = 0
        actions_taken = []
        correct_predictions = 0
        total_predictions = 0
        
        while not all(done):
            if baseline:
                actions = [baseline.compute_action(obs) for _ in range(env.num_agents)]
            elif predictive:
                predictive_agent = PredictiveAgent()
                actions = [predictive_agent.compute_action(obs) if i == 0 else trainer.compute_single_action(obs, policy_id=f"agent_{i}") for i in range(env.num_agents)]
            else:
                actions = [trainer.compute_single_action(obs, policy_id=f"agent_{i}") for i in range(env.num_agents)]
            
            # Log predictions for agent 0
            if predictive:
                predictive_agent = PredictiveAgent()
                predicted_action = predictive_agent.compute_action(obs)
                if predicted_action == actions[0]:
                    correct_predictions += 1
                total_predictions += 1
            
            obs, rewards, dones, info = env.step(actions)
            total_reward += sum(rewards)
            actions_taken.extend(actions)
        
        # Calculate cooperation rate (actions <= 2)
        coop_rate = sum(1 for a in actions_taken if a <= 2) / len(actions_taken) if actions_taken else 0
        results["coop_rate"].append(coop_rate)
        results["total_reward"].append(total_reward)
        if predictive:
            results["prediction_accuracy"].append(correct_predictions / total_predictions if total_predictions > 0 else 0)
    
    return {
        "mean_coop_rate": np.mean(results["coop_rate"]),
        "std_coop_rate": np.std(results["coop_rate"]),
        "mean_reward": np.mean(results["total_reward"]),
        "std_reward": np.std(results["total_reward"]),
        "mean_prediction_accuracy": np.mean(results["prediction_accuracy"]) if predictive else None
    }

# Main evaluation
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    
    # Load trained RL model
    config = {
        "env": "ResourceSharingEnv",
        "env_config": {"num_agents": 2, "max_resource": 10, "max_rounds": 100},
        "framework": "torch",
        "num_gpus": 0,
        "num_workers": 1,
        "multiagent": {
            "policies": {f"agent_{i}": (None, ResourceSharingEnv().observation_space, ResourceSharingEnv().action_space, {})
                         for i in range(2)},
            "policy_mapping_fn": lambda agent_id: f"agent_{agent_id}",
        },
        "model": {"fcnet_hiddens": [64, 64]},
    }
    
    trainer = PPOTrainer(config=config)
    checkpoint_path = "checkpoints/final"  # Update with actual checkpoint path
    if os.path.exists(checkpoint_path):
        trainer.restore(checkpoint_path)
    else:
        print("Checkpoint not found! Train first.")
        exit(1)
    
    env = ResourceSharingEnv(num_agents=2)
    
    # Evaluate trained policy
    trained_results = evaluate_policy(trainer, env)
    print("Trained Policy:")
    print(f"Mean Coop Rate: {trained_results['mean_coop_rate']:.2f} ± {trained_results['std_coop_rate']:.2f}")
    print(f"Mean Reward: {trained_results['mean_reward']:.2f} ± {trained_results['std_reward']:.2f}")
    
    # Evaluate random baseline
    random_results = evaluate_policy(trainer, env, baseline=RandomAgent())
    print("\nRandom Baseline:")
    print(f"Mean Coop Rate: {random_results['mean_coop_rate']:.2f} ± {random_results['std_coop_rate']:.2f}")
    print(f"Mean Reward: {random_results['mean_reward']:.2f} ± {random_results['std_reward']:.2f}")
    
    # Evaluate greedy baseline
    greedy_results = evaluate_policy(trainer, env, baseline=GreedyAgent())
    print("\nGreedy Baseline:")
    print(f"Mean Coop Rate: {greedy_results['mean_coop_rate']:.2f} ± {greedy_results['std_coop_rate']:.2f}")
    print(f"Mean Reward: {greedy_results['mean_reward']:.2f} ± {greedy_results['std_reward']:.2f}")
    
    # Evaluate predictive agent (agent 0 uses neural network)
    predictive_results = evaluate_policy(trainer, env, predictive=True)
    print("\nPredictive Agent (Agent 0):")
    print(f"Mean Coop Rate: {predictive_results['mean_coop_rate']:.2f} ± {predictive_results['std_coop_rate']:.2f}")
    print(f"Mean Reward: {predictive_results['mean_reward']:.2f} ± {predictive_results['std_reward']:.2f}")
    print(f"Mean Prediction Accuracy: {predictive_results['mean_prediction_accuracy']:.2f}")
    
    # Save evaluation results
    with open("logs/evaluation_results.txt", "w") as f:
        f.write("Trained Policy:\n")
        f.write(f"Mean Coop Rate: {trained_results['mean_coop_rate']:.2f} ± {trained_results['std_coop_rate']:.2f}\n")
        f.write(f"Mean Reward: {trained_results['mean_reward']:.2f} ± {trained_results['std_reward']:.2f}\n")
        f.write("\nRandom Baseline:\n")
        f.write(f"Mean Coop Rate: {random_results['mean_coop_rate']:.2f} ± {random_results['std_coop_rate']:.2f}\n")
        f.write(f"Mean Reward: {random_results['mean_reward']:.2f} ± {random_results['std_reward']:.2f}\n")
        f.write("\nGreedy Baseline:\n")
        f.write(f"Mean Coop Rate: {greedy_results['mean_coop_rate']:.2f} ± {greedy_results['std_coop_rate']:.2f}\n")
        f.write(f"Mean Reward: {greedy_results['mean_reward']:.2f} ± {greedy_results['std_reward']:.2f}\n")
        f.write("\nPredictive Agent (Agent 0):\n")
        f.write(f"Mean Coop Rate: {predictive_results['mean_coop_rate']:.2f} ± {predictive_results['std_coop_rate']:.2f}\n")
        f.write(f"Mean Reward: {predictive_results['mean_reward']:.2f} ± {predictive_results['std_reward']:.2f}\n")
        f.write(f"Mean Prediction Accuracy: {predictive_results['mean_prediction_accuracy']:.2f}\n")
    
    ray.shutdown()