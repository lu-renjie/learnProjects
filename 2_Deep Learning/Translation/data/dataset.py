import os.path as osp
from .tokenizer import Tokenizer


class Dataset:
    def __init__(self):
        path_english = osp.join('.', 'data', 'english.txt')
        path_chinese = osp.join('.', 'data', 'chinese.txt')
        with open(path_english, mode='r') as f:
            self.english = f.readlines()
        with open(path_chinese, mode='r') as f:
            self.chinese = f.readlines()

        assert len(self.english) == len(self.chinese)
        self.length = len(self.english)
        self.tokenizer = Tokenizer()

    def __getitem__(self, idx):
        english = self.english[idx].strip()
        english = self.tokenizer.tokenize(english)
        chinese = self.chinese[idx].strip()
        chinese = list(chinese)  # simply split characters
        return english, chinese

    def __len__(self):
        return self.length

