import yaml

from program import Program
from utils import setup_seed


if __name__ == '__main__':
    with open('config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        print('Description: ', cfg['description'])
        input('press <ENTER> to continue...')

    setup_seed(cfg['seed'])  # set seed before create program object

    p = Program(cfg)
    if cfg['train?']:
        p.train()
    else:
        # do anything you want
        pass


