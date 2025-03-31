import os
import sys
import time
import logging
import os.path as osp


def get_logger(save_dir, log=False):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s: %(message)s")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        filename = time.strftime('%Y-%m-%d_%H:%M_log.txt', time.localtime())
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
