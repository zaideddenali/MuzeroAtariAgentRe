import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from core.model import MuZeroNetwork
from core.replay_buffer import ReplayBuffer, Game
from core.mcts import MCTS
from utils.config import MuZeroConfig
import gym

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
    
    model = MuZeroNetwork(config.input_shape, config.action_space_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
    replay_buffer = ReplayBuffer(config)
    mcts = MCTS(config)
    
    env = gym.make(config.env_name)
    
    for episode in range(config.num_episodes):
        # Self-play
        game = play_game(config, model, env, mcts)
        replay_buffer.save_game(game)
        
        # Training step
        if len(replay_buffer.buffer) >= config.batch_size:
            update_weights(model, optimizer, replay_buffer, config)
            
        if episode % config.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoint_{episode}.pth")
            print(f"Episode {episode} completed. Model saved.")

def play_game(config, model, env, mcts):
    observation = env.reset()
    game = Game(config.action_space_size, config.discount)
    done = False
    
    while not done:
        obs_tensor = torch.tensor(observation).float().unsqueeze(0).to(config.device)
        action, root = mcts.search(model, obs_tensor, list(range(config.action_space_size)))
        
        next_observation, reward, done, _ = env.step(action)
        
        visits = [0] * config.action_space_size
        for a, child in root.children.items():
            visits[a] = child.visit_count
        total_visits = sum(visits)
        visits = [v / total_visits for v in visits]
        
        game.store_step(observation, action, reward, visits, root.value)
        observation = next_observation
        
    return game

def update_weights(model, optimizer, replay_buffer, config):
    batch = replay_buffer.sample_batch()
    
    total_loss = 0
    for obs, actions, targets in batch:
        obs_tensor = torch.tensor(obs).float().unsqueeze(0).to(config.device)
        hidden_state, policy_logits, value = model.initial_inference(obs_tensor)
        
        # Initial step loss
        target_value, target_reward, target_policy = targets[0]
        loss = F.mse_loss(value, torch.tensor([[target_value]]).to(config.device))
        loss += F.cross_entropy(policy_logits, torch.tensor([np.argmax(target_policy)]).to(config.device))
        
        # Unrolled steps loss
        for i in range(len(actions)):
            action = torch.tensor([actions[i]]).to(config.device)
            hidden_state, reward, policy_logits, value = model.recurrent_inference(hidden_state, action)
            
            target_value, target_reward, target_policy = targets[i+1]
            loss += F.mse_loss(value, torch.tensor([[target_value]]).to(config.device))
            loss += F.mse_loss(reward, torch.tensor([[target_reward]]).to(config.device))
            loss += F.cross_entropy(policy_logits, torch.tensor([np.argmax(target_policy)]).to(config.device))
            
        total_loss += loss
        
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

if __name__ == "__main__":
    config = MuZeroConfig()
    train(config)
