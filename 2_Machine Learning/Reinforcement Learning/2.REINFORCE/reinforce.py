import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


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
        else:
            raise 'unsupport activation faunction'
    
    def forward(self, x):
        x = self.activate_fn(self.fc1(x))
        for module in self.hidden_layers:
            x = self.activate_fn(module(x))
        x = self.fc2(x)
        return x


class REINFORCE:
    def __init__(self, _config):
        self.policy = MLP(_config)
        self.count = 0
        self.gamma = _config['gamma']
        self.optimizer = optim.Adam(self.policy.parameters(), lr=_config['lr'])
        self.trajectory = Trajectory()

    @torch.no_grad()
    def take_action(self, s, deterministic=False):
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        p = self.policy(s).squeeze()
        p = torch.softmax(p - p.max(), dim=-1)
        if deterministic:
            return p.argmax().item()
        else:
            return torch.multinomial(p, num_samples=1).item()
    
    def store(self, s, a, r):
        self.trajectory.push(s, a, r)

    def learn(self):
        s, a, G = self.trajectory.to_tensor()

        # 计算累计折扣奖励
        for i in range(2, len(self.trajectory) + 1):
            G[-i] += G[-i+1] * self.gamma

        # 学习
        for i in range(len(self.trajectory)):
            p = self.policy(s[i:i+1, :])
            p = torch.log_softmax(p, dim=1)[0, a[i]]
            loss = - G[i] * p * self.gamma**i

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # reset
        self.trajectory = Trajectory()


class Trajectory:
    def __init__(self):
        self.s = []
        self.a = []
        self.r = []
        self.length = 0

    def push(self, s, a, r):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.length += 1
    
    def to_tensor(self):
        self.s = torch.tensor(np.stack(self.s, axis=0)).float()
        self.a = torch.tensor(self.a).unsqueeze(1)
        self.r = torch.tensor(self.r).unsqueeze(1)
        return self.s, self.a, self.r

    def __len__(self):
        return self.length
