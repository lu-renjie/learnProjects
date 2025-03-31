import yaml
import torch
import random
import numpy as np
import torch.backends.cudnn

from program import Program


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    with open('config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        setup_seed(cfg['seed'])
        print('Description: ', cfg['description'])
        input('press any key to continue...')

    p = Program(cfg)
    p.train()

