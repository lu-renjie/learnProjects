import torch.nn as nn


class MLP(nn.Module):
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
        elif cfg['activation_fn'] == 'softplus':
            self.activate_fn = nn.Softplus()
        else:
            raise 'unsupport activation faunction'
    
    def forward(self, x):
        x = self.activate_fn(self.fc1(x))
        for module in self.hidden_layers:
            x = self.activate_fn(module(x))
        x = self.fc2(x)
        return x


class DuelingNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg['state_dim'], cfg['hidden_dim'])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(cfg['hidden_dim'], cfg['hidden_dim']) for _ in range(cfg['hidden_num'])]
        )
        self.V = nn.Linear(cfg['hidden_dim'], 1)
        self.A = nn.Linear(cfg['hidden_dim'], cfg['action_dim'])

        if cfg['activation_fn'] == 'relu':
            self.activate_fn = nn.ReLU()
        elif cfg['activation_fn'] == 'tanh':
            self.activate_fn = nn.Tanh()
        elif cfg['activation_fn'] == 'sigmoid':
            self.activate_fn = nn.Sigmoid()
        elif cfg['activation_fn'] == 'softplus':
            self.activate_fn = nn.Softplus()
        else:
            raise 'unsupport activation faunction'
    
    def forward(self, x):
        x = self.activate_fn(self.fc1(x))
        for module in self.hidden_layers:
            x = self.activate_fn(module(x))
        a = self.A(x)
        v = self.V(x)
        q = a + v - a.mean(dim=1, keepdim=True)
        return q

