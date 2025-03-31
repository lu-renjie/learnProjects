import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from agent import MLP, DuelingNet
from replaybuffer import N_Step_ReplayBuffer


class RainbowDQN:
    def __init__(self, cfg):
        self.count = 0
        self.gamma = cfg['gamma']
        self.n_step = cfg['n_step']
        self.batch_size = cfg['batch_size']
        self.copy_interval = cfg['copy_interval']
        self.buffer = N_Step_ReplayBuffer(self.gamma, cfg['buffer_size'], self.n_step)
        self.epsilon = cfg['epsilon_start']
        self.epsilon_decay = self.epsilon / cfg['max_step']

        self.no_double = cfg['no_double']
        if cfg['no_dueling']:
            self.Q = MLP(cfg)
        else:
            self.Q = DuelingNet(cfg)
        self.Q_target = copy.deepcopy(self.Q)
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
            return action.item()

    def learn(self):
        s, a, r, s_, terminated = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            discount = self.gamma ** self.n_step
            if self.no_double:
                target = r + discount * self.Q_target(s_).max(dim=1)[0].view(self.batch_size, 1)
            else:
                a_ = self.Q(s_).argmax(dim=1, keepdim=True)
                target = r + discount * self.Q_target(s_).gather(1, a_)
            target[terminated] = r[terminated]

        predict = self.Q(s).gather(1, a.long())
        loss = self.MSE(predict, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.count += 1
        self.epsilon -= self.epsilon_decay
        if self.count % self.copy_interval == 0:
            self.Q_target = copy.deepcopy(self.Q)
        
        return loss.item()
