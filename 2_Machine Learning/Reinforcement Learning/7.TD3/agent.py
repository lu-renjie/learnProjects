import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg['state_dim'], cfg['hidden_dim'])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(cfg['hidden_dim'], cfg['hidden_dim']) for _ in range(cfg['hidden_num'])]
        )
        self.fc2 = nn.Linear(cfg['hidden_dim'], cfg['action_dim'])

        if cfg['activation_fn'] == 'relu':
            self.activate_fn = nn.ReLU()
        elif cfg['activation_fn'] == 'tanh':
            self.activate_fn = nn.Tanh()
        elif cfg['activation_fn'] == 'sigmoid':
            self.activate_fn = nn.Sigmoid()
        else:
            raise 'unsupport activation faunction'
    
    def forward(self, x):
        x = self.activate_fn(self.fc1(x))
        for module in self.hidden_layers:
            x = self.activate_fn(module(x))
        x = self.fc2(x)
        return 2 * torch.tanh(x)


class Critic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg['state_dim'] + cfg['action_dim'], cfg['hidden_dim'])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(cfg['hidden_dim'], cfg['hidden_dim']) for _ in range(cfg['hidden_num'])]
        )
        self.fc2 = nn.Linear(cfg['hidden_dim'], 1)

        if cfg['activation_fn'] == 'relu':
            self.activate_fn = nn.ReLU()
        elif cfg['activation_fn'] == 'tanh':
            self.activate_fn = nn.Tanh()
        elif cfg['activation_fn'] == 'sigmoid':
            self.activate_fn = nn.Sigmoid()
        else:
            raise 'unsupport activation faunction'
    
    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = self.activate_fn(self.fc1(x))
        for module in self.hidden_layers:
            x = self.activate_fn(module(x))
        x = self.fc2(x)
        return x
