import torch
import torch.nn as nn
import torch.optim as optim

from configs import args, device


class Trainer:
    def __init__(self, generator, discriminator):
        self.G = generator
        self.D = discriminator

        self.loss_fn_G = nn.BCELoss()
        self.loss_fn_D = nn.BCELoss()
        self.optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr_D)

    def train_D(self, true_imgs):
        batch_size = true_imgs.shape[0]

        # train
        true_imgs = true_imgs.to(device)
        # true_imgs = true_imgs * 2 - 1
        prediction_real = self.D(true_imgs)
        labels = torch.full((batch_size, 1), 1.0).to(device)
        loss_real = self.loss_fn_D(prediction_real, labels)

        noise = torch.randn(batch_size, args.noise_dim).to(device)
        fake_imgs = self.G(noise)
        prediction_fake = self.D(fake_imgs)
        labels = torch.full((batch_size, 1), 0.0).to(device)
        loss_fake = self.loss_fn_D(prediction_fake, labels)

        loss = loss_real + loss_fake
        self.optimizer_D.zero_grad()
        loss.backward()
        self.optimizer_D.step()

        accuracy = (prediction_fake < 0.5).sum() + (prediction_real > 0.5).sum()
        accuracy = accuracy / (2 * batch_size)

        return loss.item(), accuracy

    def train_G(self):
        batch_size = args.batch_size

        noise = torch.randn(batch_size, args.noise_dim).to(device)
        fake_imgs = self.G(noise)
        labels = torch.ones(batch_size, 1).to(device)

        prediction = self.D(fake_imgs)
        loss = self.loss_fn_G(prediction, labels)

        self.optimizer_G.zero_grad()
        loss.backward()
        self.optimizer_G.step()

        return loss.item()
