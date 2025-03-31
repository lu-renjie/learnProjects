import torch
import numpy as np
import os.path as osp

from gensim.models import Word2Vec, KeyedVectors


class Preprocessor:
    def __init__(self, cfg, language, train_set):
        self.vector_size = cfg['vector_size']
        self.min_count = cfg['min_count']
        self.model_path = osp.join('.', 'data', f'wordvec_{language}.model')

        if cfg['train_word2vec']:
            self.train_word2vec(train_set)

        self.wv = KeyedVectors.load(self.model_path)
        self.PAD = self.wv.get_index('<PAD>')
        self.UNK = self.wv.get_index('<UNK>')
        self.BOS = self.wv.get_index('<BOS>')
        self.EOS = self.wv.get_index('<EOS>')

        print(language, 'word num', len(self.wv.key_to_index))

    def train_word2vec(self, train_set):
        #  train word2vec
        word2vec = Word2Vec(
            train_set,
            vector_size=self.vector_size,
            min_count=self.min_count,
            workers=1, # 多进程调度会造成随机性，需要设置为1
        )
        wv = word2vec.wv

        # add special tokens
        vectors = np.random.randn(2, self.vector_size)
        wv.add_vectors(['<UNK>', '<PAD>'], vectors)
        wv.save(self.model_path)

    def words_to_ids(self, words):
        ids = []
        for word in words:
            ids.append(self.wv.get_index(word, default=self.UNK))
        return torch.tensor(ids, dtype=torch.long)

    def ids_to_words(self, ids):
        words = []
        for id in ids:
            words.append(self.wv.index_to_key[id])
        return words
