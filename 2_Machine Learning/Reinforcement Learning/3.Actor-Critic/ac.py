import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agent import Actor, Critic


class AC(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.gamma = cfg['gamma']
        self.epoch_num = cfg['epoch_num']

        self.actor = Actor(cfg)
        self.critic = Critic(cfg)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg['lr_actor'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg['lr_critic'])

        self.MSE = nn.MSELoss()
        self.trajectory = Trajectory()
    
    def store(self, s, a, r, s_, done):
        self.trajectory.push(s, a, r, s_, done)

    @torch.no_grad()
    def take_action(self, s, deterministic=False):
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        p = self.actor(s).squeeze()
        p = torch.softmax(p - p.max(), dim=-1)
        if deterministic:
            return p.argmax().item()
        else:
            return torch.multinomial(p, num_samples=1).item()

    def learn(self):
        s, a, r, s_, done = self.trajectory.to_tensor()
        

        # train critic
        with torch.no_grad():
            target = r + self.gamma * self.critic(s_)
            target[done] = r[done]
        for _ in range(self.epoch_num):
            loss = self.MSE(self.critic(s), target)
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()
        
        # train actor
        with torch.no_grad():
            v = self.critic(s)
            v_ = self.critic(s_)
            target = r + self.gamma * v_
            target[done] = r[done]
            td_error = target - v
        for _ in range(self.epoch_num):
            p = torch.log_softmax(self.actor(s), dim=1).gather(1, a)
            loss = - torch.mean(td_error * p)
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

        # clear trajectory
        self.trajectory = Trajectory()


class Trajectory:
    def __init__(self):
        self.s = []
        self.a = []
        self.r = []
        self.s_ = []
        self.done = []
        self.length = 0

    def push(self, s, a, r, s_, done):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.s_.append(s_)
        self.done.append(done)
        self.length += 1
    
    def to_tensor(self):
        self.s = torch.tensor(np.stack(self.s, axis=0)).float()
        self.a = torch.tensor(self.a).unsqueeze(1).long()
        self.r = torch.tensor(self.r).unsqueeze(1).float()
        self.s_ = torch.tensor(np.stack(self.s_, axis=0)).float()
        self.done = torch.tensor(self.done).bool()
        return self.s, self.a, self.r, self.s_, self.done

    def __len__(self):
        return self.length
