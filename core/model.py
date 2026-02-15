import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class RepresentationNetwork(nn.Module):
    def __init__(self, input_shape, num_res_blocks, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_res_blocks)])

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)
        # Normalize hidden state to [0, 1] as per MuZero paper
        min_val = x.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        max_val = x.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        x = (x - min_val) / (max_val - min_val + 1e-5)
        return x

class DynamicsNetwork(nn.Module):
    def __init__(self, action_space_size, channels, num_res_blocks):
        super().__init__()
        self.conv1 = nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_res_blocks)])
        
        # Reward prediction head
        self.reward_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.reward_fc = nn.Linear(channels, 1) # Simplified for now

    def forward(self, hidden_state, action):
        # Action is broadcasted to match hidden state spatial dimensions
        action_plane = torch.ones_like(hidden_state[:, :1, :, :]) * action.view(-1, 1, 1, 1)
        x = torch.cat([hidden_state, action_plane], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)
        
        next_hidden_state = x
        # Normalize next hidden state
        min_val = next_hidden_state.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        max_val = next_hidden_state.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        next_hidden_state = (next_hidden_state - min_val) / (max_val - min_val + 1e-5)
        
        reward = self.reward_fc(F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1))
        return next_hidden_state, reward

class PredictionNetwork(nn.Module):
    def __init__(self, action_space_size, channels):
        super().__init__()
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 6 * 6, action_space_size) # Assuming 6x6 spatial dim after downsampling
        
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_fc = nn.Linear(1 * 6 * 6, 1)

    def forward(self, hidden_state):
        # Policy head
        p = F.relu(self.policy_conv(hidden_state))
        p = self.policy_fc(p.view(p.size(0), -1))
        
        # Value head
        v = F.relu(self.value_conv(hidden_state))
        v = self.value_fc(v.view(v.size(0), -1))
        
        return p, v

class MuZeroNetwork(nn.Module):
    def __init__(self, input_shape, action_space_size, num_res_blocks=6, channels=64):
        super().__init__()
        self.representation = RepresentationNetwork(input_shape, num_res_blocks, channels)
        self.dynamics = DynamicsNetwork(action_space_size, channels, num_res_blocks)
        self.prediction = PredictionNetwork(action_space_size, channels)

    def initial_inference(self, observation):
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)
        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state, action):
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value
