import gym
import numpy as np
from utils.env_wrappers import AtariWrapper

class Game:
    def __init__(self, env_name, action_space_size, discount):
        self.env = AtariWrapper(gym.make(env_name))
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.observations = []
        self.action_space_size = action_space_size
        self.discount = discount
        self.done = False

    def reset(self):
        observation = self.env.reset()
        self.observations.append(observation)
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.done = False
        return observation

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)
        self.observations.append(observation)
        self.history.append(action)
        self.rewards.append(reward)
        self.done = done
        return observation, reward, done

    def store_search_statistics(self, root):
        visits = [0] * self.action_space_size
        for a, child in root.children.items():
            visits[a] = child.visit_count
        total_visits = sum(visits)
        visits = [v / total_visits for v in visits]
        
        self.child_visits.append(visits)
        self.root_values.append(root.value)

    def make_observations(self, pos):
        return self.observations[pos]

    def make_targets(self, pos, num_unroll_steps, td_steps):
        targets = []
        for i in range(pos, pos + num_unroll_steps + 1):
            bootstrap_index = i + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * (self.discount ** td_steps)
            else:
                value = 0
            
            for j in range(i, min(len(self.rewards), bootstrap_index)):
                value += self.rewards[j] * (self.discount ** (j - i))
            
            if i < len(self.root_values):
                targets.append((value, self.rewards[i] if i < len(self.rewards) else 0, self.child_visits[i]))
            else:
                # Padding for terminal states
                targets.append((0, 0, [1/self.action_space_size] * self.action_space_size))
        return targets
