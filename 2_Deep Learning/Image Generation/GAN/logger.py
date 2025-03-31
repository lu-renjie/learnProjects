import os
import sys
import time
import logging
import os.path as osp

from configs import args


class Logger:
    def __init__(self):
        self.log = (args.log == 'true')
        self.save_dir = time.strftime('logs/%Y-%m-%d_%H:%M_log.txt', time.localtime())

        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s: %(message)s")
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if self.log:
            if not osp.exists(self.save_dir):
                os.makedirs(self.save_dir)
                fh = logging.FileHandler(os.path.join(self.save_dir, 'log.txt'), mode='w')
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)

    def info(self, message):
        self.logger.info(message)