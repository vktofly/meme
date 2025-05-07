import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
import numpy as np
import pandas as pd
from resource_sharing_env import ResourceSharingEnv
import os

# Register custom environment
def env_creator(env_config):
    return ResourceSharingEnv(**env_config)

register_env("ResourceSharingEnv", env_creator)

# Data collection function
def collect_data(trainer, env, num_episodes=100):
    data = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = [False] * env.num_agents
        episode_data = []
        
        while not all(done):
            # Get actions from trained RL agents
            actions = [trainer.compute_single_action(obs, policy_id=f"agent_{i}") for i in range(env.num_agents)]
            # Store state, actions, and next actions
            resource_pool = obs["resource_pool"][0]
            prev_actions = obs["prev_actions"]
            episode_data.append({
                "resource_pool": resource_pool,
                "prev_action_0": prev_actions[0],
                "prev_action_1": prev_actions[1],
                "action_0": actions[0],
                "action_1": actions[1]
            })
            # Step environment
            obs, rewards, dones, info = env.step(actions)
        
        data.extend(episode_data)
    
    # Save to CSV
    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/predictor_data.csv", index=False)
    return df

# Main
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
    df = collect_data(trainer, env, num_episodes=100)
    print(f"Collected {len(df)} data points. Saved to data/predictor_data.csv")
    
    ray.shutdown()