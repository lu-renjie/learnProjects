import os.path as osp
from tqdm import tqdm
from datetime import datetime


class Logger:
    def __init__(self, log_dir=None):
        self.format = '%Y-%m-%d %H:%M:%S: '
        self.file = None  # 用于写入文件
        self.pbar = None  # 进度条对象, progress bar
        self.time = None  # 用于记录时间

        if log_dir is not None:
            path = osp.join(log_dir, 'log.txt')
            assert osp.exists(log_dir), f'{log_dir} does not exists!'
            self.file = open(path, 'a')

    def __get_prefix(self):
        return datetime.now().strftime(self.format)

    def __seconds_to_str(self, seconds):
        seconds = int(seconds)
        minutes, seconds = seconds // 60, seconds % 60
        hours  , minutes = minutes // 60, minutes % 60
        if hours > 0:
            return f'{hours}h{minutes}m{seconds}s'
        else:
            return f'{minutes}m{seconds}s' if minutes > 0 else f'{seconds}s'

    def log(self, message):
        """
        在终端和文件内输出信息(如果指定了文件目录)
        """
        message = self.__get_prefix() + message
        print(message)
        if self.file is not None:
            print(message, file=self.file)

    def progress(self, description, message, total_step):
        """
        在终端输出进度条, 进度条满后记录到文件中
        """
        if self.pbar is None:
            self.pbar = tqdm(desc=description, total=total_step, position=0)
            self.time = datetime.now()

        self.pbar.set_postfix_str(message)
        self.pbar.update()

        if self.pbar.n == total_step:
            if self.file is not None:  # 进度完成了才会写入文件
                bar = '#' * 20
                time_consuming = (datetime.now() - self.time).total_seconds()
                time_consuming = self.__seconds_to_str(time_consuming)
                message = '%s%s 100%%|%s| %d/%d [%s, %s]' % (
                    self.__get_prefix(), description, bar,
                    self.pbar.n, total_step, time_consuming, message)
                print(message, file=self.file)
            self.pbar.close()
            self.pbar = None
