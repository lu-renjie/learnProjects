import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


from agent import Actor, Critic


class DDPG(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.gamma = cfg['gamma']
        self.MSE = nn.MSELoss()
        self.batch_size = cfg['batch_size']
        self.buffer = ReplayBuffer(cfg['buffer_size'])

        self.actor = Actor(cfg)
        self.critic = Critic(cfg)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg['lr_actor'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg['lr_critic'])

    @torch.no_grad()
    def take_action(self, s, sigma):
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        a = self.actor(s).item()
        if sigma > 0:
            a = np.random.normal(a, sigma)
        return np.array([a])
    
    def learn(self):
        s, a, r, s_, done = self.buffer.sample(self.batch_size)

        # train critic
        for params in self.critic.parameters():
            params.requires_grad = True
        with torch.no_grad():
            a_ = self.actor_target(s_)
            target = r + self.gamma * self.critic_target(s_, a_)
            target[done] = r[done]
        loss1 = self.MSE(self.critic(s, a), target)
        self.critic_optimizer.zero_grad()
        loss1.backward()
        self.critic_optimizer.step()

        # train actor
        for params in self.critic.parameters():  # freeze critic
            params.requires_grad = False
        loss2 = - self.critic(s, self.actor(s)).mean()
        self.actor_optimizer.zero_grad()
        loss2.backward()
        self.actor_optimizer.step()

        # update target networks
        self.update_target()
    
    def update_target(self):
        t = 0.999
        dict = self.actor.state_dict()
        for name, params in self.actor_target.named_parameters():
            params.data = t * params.data + (1 - t) * dict[name]
        dict = self.critic.state_dict()
        for name, params in self.critic_target.named_parameters():
            params.data = t * params.data + (1 - t) * dict[name]


class ReplayBuffer:
    def __init__(self, size):
        self.s = None
        self.a = torch.zeros(size, 1, dtype=torch.float32)
        self.r = torch.zeros(size, 1, dtype=torch.float32)
        self.s_ = None
        self.done = torch.zeros(size, 1, dtype=torch.bool)

        self.size = size
        self.count = 0
    
    def push(self, s, a, r, s_, done):
        s = s.flatten()
        s_ = s_.flatten()
        if self.s is None:
            width = len(s)
            self.s = torch.zeros(self.size, width, dtype=torch.float32)
            self.s_ = torch.zeros(self.size, width, dtype=torch.float32)

        index = self.count % self.size
        self.s[index] = torch.tensor(s, dtype=torch.float32)
        self.a[index] = torch.tensor(a, dtype=torch.float32)
        self.r[index] = torch.tensor(r, dtype=torch.float32)
        self.s_[index] = torch.tensor(s_, dtype=torch.float32)
        self.done[index] = done
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
        done = self.done[indices]
        return s, a, r, s_, done

    def __len__(self):
        return self.count
