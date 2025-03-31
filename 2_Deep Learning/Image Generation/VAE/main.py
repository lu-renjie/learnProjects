import yaml
import os.path as osp
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from utils import setup_seed, Logger, dict_to_markdown_table
from model import VAE


def get_log_dir(cfg):
    time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    dir1 = f"MLP_{cfg['encoder_hidden_dims']}_{cfg['decoder_hidden_dims']}_{cfg['latent_dim']}"
    dir2 = f"{time}_{cfg['description']}"
    log_dir = osp.join('log', dir1, dir2)
    return log_dir


def train(cfg, model):
    dataset = MNIST('/Users/lurenjie/Documents/dataset', transform=ToTensor())
    dataloader = DataLoader(dataset, cfg['batch_size'], shuffle=True)

    log_dir = get_log_dir(cfg)
    writer = SummaryWriter(log_dir)
    logger = Logger(log_dir)
    writer.add_text('hyperparameters', dict_to_markdown_table(cfg))

    for epoch in range(cfg['epoch_num']):
        loss_dict = model.train_epoch(dataloader)

        logger.log(str(loss_dict))
        writer.add_scalars('loss', loss_dict, epoch)

        imgs = model.sample_batch(64)
        img_grid = make_grid(imgs, nrow=8)
        path = osp.join(log_dir, f'generated_img{epoch}.png')
        save_image(img_grid, path)
    model.save(log_dir)


def scatter_imgs(x, y, images, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
 
    x, y = np.atleast_1d(x, y)

    artists = []
    for i, (x0, y0) in enumerate(zip(x, y)):
        im = OffsetImage(images[i], zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()

    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    ax.set_xticks([])
    ax.set_yticks([])
    return artists


def visualize(cfg, model):
    log_dir = cfg['trained_model_dir']
    if cfg['load_trained_model']:
        model.load(log_dir)

    if model.latent_dim == 1:
        N = 50
        z = np.random.randn(N)
        z.sort()
        imgs = model.sample_batch(z=z.reshape(N, 1))
        imgs = 1 - imgs  # 黑底变白底
        img = make_grid(imgs, nrow=N)
        save_image(img, osp.join(log_dir, 'visualization.png'))
    if model.latent_dim == 2:
        N = 20
        edge = 1.5
        x = np.linspace(-edge, edge, N)
        y = np.linspace(-edge, edge, N)
        x, y = np.meshgrid(x, y)
        z = np.stack([x.flatten(), y.flatten()], axis=1)
        # z = np.random.randn(N * N, 2)
        imgs = model.sample_batch(z=z).permute(0, 2, 3, 1).numpy()  # (B, H, W, C)
        imgs = 1 - imgs  # 黑底变白底

        # to RGBA, 把白色底变为透明的
        alpha = 1 - (imgs == 1.0).all(axis=3)[..., np.newaxis]
        imgs = np.concatenate([imgs, alpha], axis=3)

        plt.figure(figsize=(8, 6))
        scatter_imgs(z[:, 0], z[:, 1], imgs, zoom=0.5)
        plt.savefig(osp.join(log_dir, 'visualization.png'), bbox_inches='tight', pad_inches = -0.1)
        plt.show()


if __name__ == '__main__':
    with open('config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        print('Description: ', cfg['description'])
        input('press <ENTER> to continue...')

    setup_seed(cfg['seed'])  # set seed before create program object

    model = VAE(cfg)
    # train(cfg, model)
    visualize(cfg, model)
