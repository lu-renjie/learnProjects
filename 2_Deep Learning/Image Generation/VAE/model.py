import tqdm
import torch
import numpy as np
import pandas as pd
import os.path as osp
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, intput_dim, hidden_dims, out_dim):
        super().__init__()
        self.fc_list = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                fc = nn.Linear(intput_dim, hidden_dim)
            else:
                fc = nn.Linear(last_hidden_dim, hidden_dim)
            last_hidden_dim = hidden_dim
            self.fc_list.append(fc)
        self.fc_mu = nn.Linear(last_hidden_dim, out_dim)
        self.fc_sigma = nn.Linear(last_hidden_dim, out_dim)
    
    def forward(self, x):
        for fc in self.fc_list:
            x = fc(x)
            x = torch.relu(x)
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        sigma = torch.exp(sigma)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, intput_dim, hidden_dims, out_dim):
        super().__init__()
        self.fc_list = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                fc = nn.Linear(intput_dim, hidden_dim)
            else:
                fc = nn.Linear(last_hidden_dim, hidden_dim)
            last_hidden_dim = hidden_dim
            self.fc_list.append(fc)
        self.out = nn.Linear(last_hidden_dim, out_dim)
    
    def forward(self, x):
        for fc in self.fc_list:
            x = fc(x)
            x = torch.relu(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x


class Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.var = cfg['sigma'] ** 2
        self.MSELoss = nn.MSELoss()
        self.L1Loss = nn.L1Loss()    
    
    def forward(self, y, y_target):
        # loss = self.MSELoss(y, y_target)
        # loss = self.L1Loss(y, y_target)
        loss = 0.5 * self.MSELoss(y, y_target) + 0.5 * self.L1Loss(y, y_target)
        return loss / self.var


class VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.L = cfg['L']
        self.sigma = cfg['sigma']
        self.latent_dim = cfg['latent_dim']
    
        self.encoder = Encoder(28 * 28, cfg['encoder_hidden_dims'], self.latent_dim)
        self.decoder = Decoder(self.latent_dim, cfg['decoder_hidden_dims'], 28 * 28)

        self.loss_fn = Loss(cfg)
        self.optimizer = optim.AdamW(self.parameters(), lr=cfg['lr'])
    
    def train_epoch(self, dataloader):
        loss_dict = pd.Series({
            'AE_loss': 0,  # autoencoder loss
            'mu_loss': 0,  # loss of mu
            'sigma_loss': 0,  # loss of sigma
            'sum': 0,
        })

        pbar = tqdm.tqdm(desc='TRAIN', total=len(dataloader))
        for i, (img, label) in enumerate(dataloader):
            batch_size = img.shape[0]

            img = img.reshape(batch_size, 28 * 28)
            mu, sigma = self.encoder(img)

            loss1 = [None] * self.L
            for i in range(self.L):
                z = mu + sigma * torch.randn_like(sigma)
                img_hat = self.decoder(z)
                loss1[i] = self.loss_fn(img_hat, img)
            loss1 = sum(loss1) / (self.L)

            var = sigma * sigma
            loss2 = (mu * mu).sum(dim=1).mean()
            loss3 = (var - var.log() - 1).sum(dim=1).mean()

            loss = loss1 + (loss2 + loss3)

            loss_dict['AE_loss'] += loss1.detach().numpy()
            loss_dict['mu_loss'] += loss2.detach().numpy()
            loss_dict['sigma_loss'] += loss3.detach().numpy()
            loss_dict['sum'] += loss.detach().numpy()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pbar.set_postfix_str(f"loss:{loss_dict['AE_loss'] / (i+1): .2f}")
            pbar.update()
        loss_dict /= (i + 1)
        loss_dict = loss_dict.round(2).to_dict()
        return loss_dict

    @torch.no_grad()
    def sample_batch(self, batch_size=None, z=None):
        if z is None:
            z = torch.randn(batch_size, self.latent_dim)
        else:
            batch_size = len(z)
            z = torch.tensor(z, dtype=torch.float32)
        imgs = self.decoder(z)
        imgs = imgs.reshape(batch_size, 28, 28)
        imgs = torch.stack([imgs, imgs, imgs], dim=1)  # to RGB, (B, 3, 28, 28)
        return imgs

    @torch.no_grad()
    def sample(self, z=None):
        img = self.sample_batch(1, z).squeeze(0)
        return img

    def save(self, dir):
        path = osp.join(dir, 'model.pt')
        torch.save(self.state_dict(), path)
    
    def load(self, dir):
        path = osp.join(dir, 'model.pt')
        state_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(state_dict)
