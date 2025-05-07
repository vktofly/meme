import gym
import numpy as np
from gym.spaces import Discrete, Box, Dict, Tuple, MultiDiscrete

class ResourceSharingEnv(gym.Env):
    """Custom Gym environment for a multi-agent resource-sharing game."""
    def __init__(self, num_agents=2, max_resource=10, max_rounds=100):
        super(ResourceSharingEnv, self).__init__()
        self.num_agents = num_agents
        self.max_resource = max_resource
        self.max_rounds = max_rounds
        self.current_round = 0
        self.resource_pool = max_resource

        # Action space: Each agent chooses amount to take (0 to 5 units)
        self.action_space = Tuple([Discrete(6)] * num_agents)  # 0, 1, 2, 3, 4, 5

        # Observation space: Resource pool size + previous actions of all agents
        self.observation_space = Dict({
            "resource_pool": Box(low=0, high=max_resource, shape=(1,), dtype=np.float32),
            "prev_actions": Tuple([Discrete(6)] * num_agents)
        })

        # Initialize state
        self.prev_actions = [0] * num_agents

    def reset(self):
        """Reset the environment to initial state."""
        self.current_round = 0
        self.resource_pool = self.max_resource
        self.prev_actions = [0] * self.num_agents
        return self._get_obs()

    def _get_obs(self):
        """Return current observation."""
        return {
            "resource_pool": np.array([self.resource_pool], dtype=np.float32),
            "prev_actions": tuple(self.prev_actions)
        }

    def step(self, actions):
        """Execute one step in the environment."""
        assert len(actions) == self.num_agents, "Actions must match number of agents"
        self.current_round += 1

        # Calculate total demand
        total_demand = sum(actions)

        # Compute rewards
        rewards = [0] * self.num_agents
        if total_demand <= self.resource_pool:
            # Agents get what they requested
            for i in range(self.num_agents):
                rewards[i] = actions[i]
            # Cooperation bonus if all take sustainable amounts (e.g., <= 2 each)
            if all(a <= 2 for a in actions):
                rewards = [r + 1 for r in rewards]
        else:
            # Overuse penalty: Scale rewards based on available resources
            scale = self.resource_pool / total_demand if total_demand > 0 else 0
            for i in range(self.num_agents):
                rewards[i] = actions[i] * scale * 0.5  # Penalty for overuse

        # Update resource pool
        self.resource_pool = max(0, self.resource_pool - total_demand)
        self.prev_actions = actions

        # Check termination
        done = self.resource_pool <= 0 or self.current_round >= self.max_rounds
        dones = [done] * self.num_agents

        # Info dictionary (optional, for debugging)
        info = {i: {} for i in range(self.num_agents)}

        return self._get_obs(), rewards, dones, info

    def render(self, mode="human"):
        """Render the current state (for debugging)."""
        print(f"Round {self.current_round}: Pool={self.resource_pool}, Actions={self.prev_actions}")

# This is a test.