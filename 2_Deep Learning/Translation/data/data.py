import torch
import pickle
import random
import os.path as osp
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .dataset import Dataset
from .preprocessor import Preprocessor


class Data:
    def __init__(self, cfg):
        dataset_path = osp.join('.', 'data', 'dataset.pkl')
        if cfg['load_pickle_dataset']:
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
        else:
            dataset = Dataset()
            dataset = [(['<BOS>'] + en + ['<EOS>'], ['<BOS>'] + cn + ['<EOS>']) for en, cn in dataset]
            random.shuffle(dataset)
            with open(dataset_path, 'wb') as f:
                pickle.dump(dataset, f)

        self.batch_size = cfg['batch_size']

        self.test_set_size = cfg['test_set_size']
        self.train_set_size = len(dataset) - self.test_set_size
        self.train_set = dataset[:self.train_set_size]
        self.test_set = dataset[self.train_set_size:]

        english_train_set = [en for en, cn in self.train_set]
        chinese_train_set = [cn for en, cn in self.train_set]
        self.processor_english = Preprocessor(cfg, 'english', english_train_set)
        self.processor_chinese = Preprocessor(cfg, 'chinese', chinese_train_set)

        #  convert words to ids
        self.train_set = [
            (self.processor_english.words_to_ids(en),
             self.processor_chinese.words_to_ids(cn))
            for en, cn in self.train_set]
        self.test_set = [
            (self.processor_english.words_to_ids(en),
             self.processor_chinese.words_to_ids(cn))
            for en, cn in self.test_set]

    def collate_fn(self, batch):
        english, chinese = zip(*batch)
        actual_length_en = torch.tensor([len(x) for x in english])
        actual_length_cn = torch.tensor([len(x) for x in chinese])
        english = pad_sequence(english, batch_first=True, padding_value=self.processor_english.PAD)
        chinese = pad_sequence(chinese, batch_first=True, padding_value=self.processor_chinese.PAD)
        return english, actual_length_en, chinese, actual_length_cn

    def get_train_loader(self):
        dataloader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn)
        return dataloader

    def get_test_set(self):
        return self.test_set

    def get_vectors_english(self):
        vectors = self.processor_english.wv.vectors
        return torch.tensor(vectors)

    def get_vectors_chinese(self):
        vectors = self.processor_chinese.wv.vectors
        return torch.tensor(vectors)

    def get_special_tokens_english(self):
        d = {
            'PAD': self.processor_english.PAD,
            'UNK': self.processor_english.UNK,
            'BOS': self.processor_english.BOS,
            'EOS': self.processor_english.EOS,
        }
        return d

    def get_special_tokens_chinese(self):
        d = {
            'PAD': self.processor_chinese.PAD,
            'UNK': self.processor_chinese.UNK,
            'BOS': self.processor_chinese.BOS,
            'EOS': self.processor_chinese.EOS,
        }
        return d

    def ids_to_words_english(self, ids):
        return self.processor_english.ids_to_words(ids)

    def ids_to_words_chinese(self, ids):
        return self.processor_chinese.ids_to_words(ids)

    def sentence_to_ids_chinese(self, sentence):
        words = ['<BOS>'] + list(sentence) + ['<EOS>']
        return self.processor_chinese.words_to_ids(words)

    def verbose(self):
        total_size = self.train_set_size + self.test_set_size
        train_set_mean_length = sum([len(en) for en, cn in self.train_set]) / self.train_set_size
        test_set_mean_length = sum([len(en) for en, cn in self.test_set]) / self.test_set_size
        print('--------------------------------')
        print('train set size: %6d' % (self.train_set_size),
              'average english sentence length: %.2f' % (train_set_mean_length - 2))
        print('test  set size: %6d' % (self.test_set_size),
              'average english sentence length: %.2f' % (test_set_mean_length - 2))
        print('total     size: %6d' % (total_size))
        print('--------------------------------')
