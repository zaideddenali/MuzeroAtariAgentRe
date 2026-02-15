import math
import numpy as np
import torch

class Node:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, policy_logits, hidden_state, reward):
        self.hidden_state = hidden_state
        self.reward = reward
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        for action in actions:
            self.children[action] = Node(policy[action])

class MCTS:
    def __init__(self, config):
        self.config = config

    def search(self, model, observation, actions):
        with torch.no_grad():
            hidden_state, policy_logits, value = model.initial_inference(observation)
        
        root = Node(0)
        root.expand(actions, policy_logits, hidden_state, 0)
        self.add_exploration_noise(root)

        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]

            while node.children and any(child.visit_count > 0 for child in node.children.values()):
                action, node = self.select_child(node)
                search_path.append(node)

            parent = search_path[-2] if len(search_path) > 1 else None
            if parent:
                action = [a for a, n in parent.children.items() if n == node][0]
                with torch.no_grad():
                    hidden_state, reward, policy_logits, value = model.recurrent_inference(
                        parent.hidden_state, torch.tensor([action]).to(self.config.device)
                    )
                node.expand(actions, policy_logits, hidden_state, reward.item())
            
            self.backpropagate(search_path, value.item())

        return self.select_action(root), root

    def select_child(self, node):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            score = self.ucb_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def ucb_score(self, parent, child):
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        
        prior_score = pb_c * child.prior
        value_score = child.value # Simplified; MuZero uses min-max normalization
        return prior_score + value_score

    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + self.config.discount * value

    def select_action(self, root):
        visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
        # During training, sample based on visit counts; during eval, take max
        actions, counts = zip(*visit_counts)
        return actions[np.argmax(counts)]

    def add_exploration_noise(self, node):
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
        frac = self.config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac
