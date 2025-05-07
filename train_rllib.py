import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
import gym
import numpy as np
import os
from resource_sharing_env import ResourceSharingEnv

# Register custom environment
def env_creator(env_config):
    return ResourceSharingEnv(**env_config)

register_env("ResourceSharingEnv", env_creator)

# RLlib configuration
config = {
    "env": "ResourceSharingEnv",
    "env_config": {
        "num_agents": 2,
        "max_resource": 10,
        "max_rounds": 100
    },
    "framework": "torch",
    "num_gpus": 0,
    "num_workers": 2,
    "multiagent": {
        "policies": {
            f"agent_{i}": (None, ResourceSharingEnv().observation_space, ResourceSharingEnv().action_space, {})
            for i in range(2)
        },
        "policy_mapping_fn": lambda agent_id: f"agent_{agent_id}",
    },
    "model": {
        "fcnet_hiddens": [64, 64],
    },
    "lr": 5e-4,
    "gamma": 0.99,
    "rollout_fragment_length": 200,
    "train_batch_size": 4000,
}

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Create directory for checkpoints and logs
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Custom callback to log cooperation metrics
class CoopCallback(ray.rllib.algorithms.callbacks.DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.coop_rates = []
        self.episode_rewards = []

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        # Calculate cooperation rate (actions <= 2 are cooperative)
        actions = episode.last_info_for(0).get("actions", [])
        if actions:
            coop_rate = sum(1 for a in actions if a <= 2) / len(actions)
            self.coop_rates.append(coop_rate)
            episode.custom_metrics["coop_rate"] = coop_rate
        # Log episode reward
        reward = episode.total_reward
        self.episode_rewards.append(reward)
        episode.custom_metrics["total_reward"] = reward

# Train with PPO
trainer = PPOTrainer(config=config)
num_iterations = 100  # Train for 100 iterations
log_file = open("logs/training_log.csv", "w")
log_file.write("iteration,mean_reward,coop_rate\n")

for i in range(num_iterations):
    result = trainer.train()
    mean_reward = result["episode_reward_mean"]
    coop_rate = result.get("custom_metrics", {}).get("coop_rate_mean", 0)
    print(f"Iteration {i}: Mean Reward = {mean_reward:.2f}, Coop Rate = {coop_rate:.2f}")
    log_file.write(f"{i},{mean_reward},{coop_rate}\n")
    
    # Save checkpoint every 10 iterations
    if i % 10 == 0:
        checkpoint_path = trainer.save("checkpoints")
        print(f"Checkpoint saved at {checkpoint_path}")

log_file.close()

# Save final model
final_checkpoint = trainer.save("checkpoints/final")
print(f"Final checkpoint saved at {final_checkpoint}")

# Shutdown Ray
ray.shutdown()
#this is a test