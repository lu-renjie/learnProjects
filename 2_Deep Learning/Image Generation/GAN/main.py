import torch
import os.path as osp

from data import get_dataloaders
from model import Generator, Discriminator
from train import Trainer
from logger import Logger
from configs import args, device
from torchvision.utils import save_image


def main():
    train_loader = get_dataloaders(args.root, batch_size=args.batch_size)
    generator = Generator(args.noise_dim).to(device)
    discriminator = Discriminator().to(device)

    logger = Logger()
    for k, v in vars(args).items():
        logger.info(f'{k} = {v}')

    trainer = Trainer(generator, discriminator)
    step = 0
    for epoch in range(args.epoch_num):
        for imgs, _ in train_loader:
            step += 1

            loss_D, accuracy = trainer.train_D(imgs)
            loss_G = trainer.train_G()

            interval = 50
            if step % interval == 0:
                logger.info(f'epoch[{epoch}/{args.epoch_num}], loss_D: {loss_D:.4e}, accuracy: {accuracy:.4f}')
                logger.info(f'epoch[{epoch}/{args.epoch_num}], loss_G: {loss_G:.4e}')
                fake_imgs = generator(torch.randn(16, args.noise_dim).to(device))
                if logger.log:
                    save_image(fake_imgs, osp.join(logger.save_dir, f'images_{step}.png'))
    if logger.log:
        torch.save(generator, osp.join(logger.save_dir, 'generator.pth'))
        torch.save(discriminator, osp.join(logger.save_dir, 'discriminator.pth'))
    logger.info('finished training.')


if __name__ == '__main__':
    main()
