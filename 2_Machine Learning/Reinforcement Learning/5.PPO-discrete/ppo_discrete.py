import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.data.sampler import BatchSampler, RandomSampler

from normalization import RewardScaling
from networks import ActorDiscrete, Critic


def compute_GAE(td_error, gamma, lmbda, terminated):
    advantages = torch.zeros_like(td_error)

    last_GAE = 0
    for i in reversed(range(len(td_error))):
        if terminated[i]:
            advantages[i] = td_error[i]
        else:
            advantages[i] = td_error[i] + gamma * lmbda * last_GAE
        last_GAE = advantages[i]

    return advantages


class PPO(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.gamma = cfg['gamma']
        self.eps = cfg['clip_eps']
        self.lmbda = cfg['GAE_lambda']
        self.max_norm = cfg['max_norm']
        self.num_envs = cfg['num_envs']
        self.epoch_num = cfg['epoch_num']
        self.batch_size = cfg['batch_size']
        self.loss_weight = cfg['loss_weight']
        self.normalize_GAE = cfg['normalize_GAE']
        self.use_clipped_loss = cfg['use_clipped_loss']

        self.actor = ActorDiscrete(cfg)
        self.critic = Critic(cfg)

        self.MSELoss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=cfg['lr'], eps=cfg['adam_eps'])
        self.scheduler = Scheduler(self.optimizer, cfg['max_step'] // (cfg['num_envs'] * cfg['sample_num']))

        self.buffer = Buffer(cfg)
    
    @torch.no_grad()
    def take_action_test(self, s):  # no batch and deterministic
        s = torch.tensor(s).unsqueeze(0)
        probs = self.actor(s).squeeze()
        return probs.argmax().item()

    @torch.no_grad()
    def take_action_train(self, s):  # batch and random
        s = torch.tensor(s)
        probs = self.actor(s)
        action = torch.multinomial(probs, num_samples=1)
        return action.squeeze().numpy(), probs.gather(1, action).squeeze()
    
    @torch.no_grad()
    def prepare_data(self):
        s, a, r, s_, terminated, old_probs = self.buffer.get_data()
        
        target = torch.zeros_like(r)
        advantages = torch.zeros_like(r)

        # compute GAE and target of each trajectory
        for i in range(self.num_envs):
            v = self.critic(s[i]).squeeze()
            v_ = self.critic(s_[i]).squeeze()
            td_error = r[i] + self.gamma * v_ * (1 - terminated[i].float()) - v
            advantages[i] = compute_GAE(td_error, self.gamma, self.lmbda, terminated[i])
            # advantages[i] = td_error
            target[i] = advantages[i] + v

        # (N, M, ...) to (N * M, ...)
        size = a.numel()
        s = s.reshape(size, -1)
        a = a.reshape(size, 1)
        target = target.reshape(size, 1)
        old_probs = old_probs.reshape(size, 1)
        advantages = advantages.reshape(size, 1)

        return s, a, target, old_probs, advantages
        
    def learn(self):
        s, a, target, old_probs, advantages = self.prepare_data()

        # train several epochs
        for _ in range(self.epoch_num):
            sampler = RandomSampler(range(len(target)))
            for batch_idx in BatchSampler(sampler, self.batch_size, False):
                # value loss
                value_loss = self.MSELoss(self.critic(s[batch_idx]), target[batch_idx])

                # policy loss
                probs = self.actor(s[batch_idx])
                ratio = probs.gather(1, a[batch_idx]) / old_probs[batch_idx]
                adv_batch = advantages[batch_idx]
                if self.normalize_GAE:
                    adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)
                if self.use_clipped_loss:
                    cliped_ratio = torch.clip(ratio, min=1-self.eps, max=1+self.eps)
                    policy_loss = - torch.min(adv_batch * ratio, adv_batch * cliped_ratio).mean()
                else:
                    policy_loss = - torch.mean(adv_batch * ratio)

                # entropy loss
                entropy_loss = - Categorical(probs).entropy().mean()

                # total loss
                loss = self.loss_weight['value']   * value_loss  + \
                       self.loss_weight['policy']  * policy_loss + \
                       self.loss_weight['entropy'] * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)
                self.optimizer.step()

                # approximate KL divergence to debug
                approx_kl = ((ratio - 1) - ratio.log()).mean()
        self.scheduler.step()
        return value_loss.item(), policy_loss.item(), entropy_loss.item(), approx_kl.item()


class Buffer:
    def __init__(self, cfg):
        self.N = cfg['num_envs']
        self.M = cfg['sample_num']

        self.use_reward_scaling = cfg['use_reward_scaling']
        if self.use_reward_scaling:
            self.reward_scaling = RewardScaling(shape=1, gamma=cfg['gamma'])

        self.count = 0
        self.s = torch.zeros(self.N, self.M, cfg['state_dim'], dtype=torch.float32)
        self.a = torch.zeros(self.N, self.M, dtype=torch.int64)
        self.r = torch.zeros(self.N, self.M, dtype=torch.float32)
        self.s_ = torch.zeros(self.N, self.M, cfg['state_dim'], dtype=torch.float32)
        self.terminated = torch.zeros(self.N, self.M, dtype=torch.bool)
        self.probs = torch.zeros(self.N, self.M, dtype=torch.float32)

    def push(self, s, a, r, s_, terminated, probs):
        if self.use_reward_scaling:
            r = self.reward_scaling(r)

        self.s[:, self.count, :] = torch.from_numpy(s)
        self.a[:, self.count] = torch.from_numpy(a)
        self.r[:, self.count] = torch.from_numpy(r)
        self.s_[:, self.count, :] = torch.from_numpy(s_)
        self.terminated[:, self.count] = torch.from_numpy(terminated)
        self.probs[:, self.count] = probs
        self.count += 1
    
    def reset(self):
        self.count = 0

    def get_data(self):
        return self.s, self.a, self.r, self.s_, self.terminated, self.probs


class Scheduler:
    def __init__(self, optimizer, total_step):
        self.optimizer = optimizer
        self.base_lr = optimizer.param_groups[0]['lr']

        self.total_step = total_step
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        for p in self.optimizer.param_groups:
            p['lr'] = self.base_lr * (1 - self.current_step / self.total_step)
