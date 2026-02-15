import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, config):
        self.config = config
        self.buffer = []
        self.max_size = config.window_size

    def save_game(self, game):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self):
        games = random.choices(self.buffer, k=self.config.batch_size)
        game_pos = [(g, random.randint(0, len(g.history) - 1)) for g in games]
        
        batch = []
        for game, pos in game_pos:
            # Sample unrolled steps for training
            observations = game.make_observations(pos)
            actions = game.history[pos:pos + self.config.num_unroll_steps]
            targets = game.make_targets(pos, self.config.num_unroll_steps, self.config.td_steps)
            batch.append((observations, actions, targets))
        
        return batch

class Game:
    def __init__(self, action_space_size, discount):
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.observations = []
        self.action_space_size = action_space_size
        self.discount = discount

    def store_step(self, observation, action, reward, visits, root_value):
        self.observations.append(observation)
        self.history.append(action)
        self.rewards.append(reward)
        self.child_visits.append(visits)
        self.root_values.append(root_value)

    def make_observations(self, pos):
        # Return the observation at pos
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
