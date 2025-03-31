import copy
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


class DQN:
    def __init__(self, cfg):
        self.Q = MLP(cfg)
        self.Q_target = copy.deepcopy(self.Q)

        self.count = 0
        self.gamma = cfg['gamma']
        self.batch_size = cfg['batch_size']
        self.copy_interval = cfg['copy_interval']
        self.buffer = ReplayBuffer(cfg['buffer_size'])
        self.epsilon = cfg['epsilon_start']

        self.epsilon_decay = self.epsilon / cfg['max_step']

        self.optimizer = optim.Adam(self.Q.parameters(), lr=cfg['lr'])
        self.MSE = nn.MSELoss()

    @torch.no_grad()
    def take_action(self, s, deterministic=False):
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        scores = self.Q(s).squeeze()
        action = scores.argmax()
        if deterministic:
            return action.item()
        elif np.random.rand() < self.epsilon:  # epsilon greedy
            return np.random.choice(len(scores))
        else:
            return action.numpy()

    def learn(self):
        s, a, r, s_, terminated = self.buffer.sample(self.batch_size)
        with torch.no_grad():
            target = r + self.gamma * self.Q_target(s_).max(dim=1)[0].view(self.batch_size, 1)
            target[terminated] = r[terminated]  # 结束状态的回报单独算, 这一步很重要, 因为是准确的标签
        predict = self.Q(s).gather(1, a.long())
        loss = self.MSE(predict, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.count += 1
        self.epsilon -= self.epsilon_decay
        if self.count % self.copy_interval == 0:
            self.Q_target = copy.deepcopy(self.Q)


class ReplayBuffer:
    def __init__(self, size):
        self.s = None
        self.a = torch.zeros(size, 1, dtype=torch.float32)
        self.r = torch.zeros(size, 1, dtype=torch.float32)
        self.s_ = None
        self.terminated = torch.zeros(size, 1, dtype=torch.bool)

        self.size = size
        self.count = 0
    
    def push(self, s, a, r, s_, terminated):
        if self.s is None:
            width = len(s)
            self.s = torch.zeros(self.size, width, dtype=torch.float32)
            self.s_ = torch.zeros(self.size, width, dtype=torch.float32)

        index = self.count % self.size
        self.s[index] = torch.tensor(s, dtype=torch.float32)
        self.a[index] = torch.tensor(a, dtype=torch.float32)
        self.r[index] = torch.tensor(r, dtype=torch.float32)
        self.s_[index] = torch.tensor(s_, dtype=torch.float32)
        self.terminated[index] = terminated
        self.count += 1
    
    def full(self):
        return self.size <= self.count

    def sample(self, batch_size):
        max_idx = min(self.size, self.count)
        indices = np.random.choice(max_idx, batch_size, replace=False)
        s = self.s[indices]
        a = self.a[indices]
        r = self.r[indices]
        s_ = self.s_[indices]
        terminated = self.terminated[indices]
        return s, a, r, s_, terminated

    def __len__(self):
        return self.count
