import torch
import torch.nn as nn


def orthogonal_init(m, gain):
    nn.init.orthogonal_(m.weight, gain=gain)
    nn.init.constant_(m.bias, val=0)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg['state_dim'], cfg['hidden_dim'])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(cfg['hidden_dim'], cfg['hidden_dim']) for _ in range(cfg['hidden_num'])])

        # 激活函数设置
        if cfg['activation_fn'] == 'relu':
            self.activate_fn = nn.ReLU()
        elif cfg['activation_fn'] == 'tanh':
            self.activate_fn = nn.Tanh()
        else:
            raise 'unsupport activation function'
        
        # 初始化方法设置
        if cfg['init_method'] == 'default':
            pass
        elif cfg['init_method'] == 'orthogonal':
            orthogonal_init(self.fc1, gain=1)
            for module in self.hidden_layers:
                orthogonal_init(module, gain=1)
        else:
            raise 'unsupport initialization method'
    
    def forward(self, x):
        x = self.activate_fn(self.fc1(x))
        for fc in self.hidden_layers:
            x = self.activate_fn(fc(x))
        return x


class ActorDiscrete(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = MLP(cfg)
        self.fc = nn.Linear(cfg['hidden_dim'], cfg['action_dim'])
        
        # 初始化方法设置
        if cfg['init_method'] == 'default':
            pass
        elif cfg['init_method'] == 'orthogonal':
            orthogonal_init(self.fc, gain=0.01)
        else:
            raise 'unsupport initialization method'

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        probs = torch.softmax(x - x.max(), dim=-1)
        return probs


class Critic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = MLP(cfg)
        self.fc = nn.Linear(cfg['hidden_dim'], 1)

        # 初始化方法设置
        if cfg['init_method'] == 'default':
            pass
        elif cfg['init_method'] == 'orthogonal':
            orthogonal_init(self.fc, gain=1)
        else:
            raise 'unsupport initialization method'

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x
