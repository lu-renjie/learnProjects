import torch
import numpy as np


class N_Step_ReplayBuffer:
    def __init__(self, gamma, size, n_step):
        self.s = None
        self.a = torch.zeros(size, 1, dtype=torch.float32)
        self.R = torch.zeros(size, 1, dtype=torch.float32)
        self.s_N = None
        self.terminated = torch.zeros(size, 1, dtype=torch.bool)

        self.size = size
        self.n_step = n_step
        self.step_count = 0
        self.gamma = gamma
        self.count = 0
    
    def push(self, s, a, r, s_, terminated):
        if self.s is None:
            width = len(s)
            self.s = torch.zeros(self.size, width, dtype=torch.float32)
            self.s_N = torch.zeros(self.size, width, dtype=torch.float32)

        self.step_count += 1

        index = self.count % self.size
        if self.step_count == 1:
            self.s[index] = torch.tensor(s, dtype=torch.float32)
            self.a[index] = torch.tensor(a, dtype=torch.float32)
            self.R[index] = 0
        if self.step_count <= self.n_step:
            discount = self.gamma ** (self.step_count - 1)
            self.R[index] += discount * torch.tensor(r, dtype=torch.float32)
        if (self.step_count == self.n_step) or terminated:
            self.s_N[index] = torch.tensor(s_, dtype=torch.float32)
            self.terminated[index] = terminated
            self.count += 1
            self.step_count = 0

    def sample(self, batch_size):
        max_idx = min(self.size, self.count)
        indices = np.random.choice(max_idx, batch_size, replace=False)
        s = self.s[indices]
        a = self.a[indices]
        R = self.R[indices]
        s_N = self.s_N[indices]
        terminated = self.terminated[indices]
        return s, a, R, s_N, terminated

    def __len__(self):
        return self.count
