import torch
import os.path as osp
import torch.nn as nn
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = Embedding(
            input_dim,
            freeze=False,
            cache_dir='/Users/lurenjie/Documents/4_pretrained/glove.6B'
        )
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, texts):
        batch_size = len(texts)
        h0 = self.h0.expand(-1, batch_size, -1)
        c0 = self.h0.expand(-1, batch_size, -1)

        sequence, lengths, pad_mask = self.embedding(texts)
        sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        sequence, (h, c) = self.rnn(sequence, (h0, c0))
        sequence = pad_packed_sequence(sequence, batch_first=True)
        
        h = h.permute(1, 0, 2).reshape(batch_size, -1)
        logits = self.classifier(h)
        return logits


def preprocess(sentence):
    """
    Args:
        sentence: str
    
    Returns:
        list[str]
    """
    sentence = sentence.lower()
    tokens = word_tokenize(sentence)
    return tokens


class Embedding(nn.Module):
    """
    cache_dir is 'xxxx/glove.6B', downloaded from torchtext GloVe
    """
    def __init__(self, vector_size, freeze, cache_dir):
        super().__init__()
        path = osp.join(cache_dir, f'glove.6B.{vector_size}d.txt.pt')
        self.itos, self.stoi, vectors, vector_size = torch.load(path)

        # add <PAD> and <UNK>
        PAD_UNK = torch.zeros(2, vector_size)
        vectors = torch.cat([PAD_UNK, vectors], dim=0)
        # update itos
        self.itos.insert(0, '<UNK>')
        self.itos.insert(0, '<PAD>')  # ['<PAD>', '<UNK>', ...]
        # update stoi
        for s in self.stoi:
            self.stoi[s] += 2
        self.stoi['<PAD>'] = 0
        self.stoi['<UNK>'] = 1

        self.embedding = nn.Embedding.from_pretrained(vectors, padding_idx=0, freeze=freeze)

    def forward(self, texts):
        """
        Args:
            texts: list[str], a batch of text
        Returns:
            a tensor of shape (B, L, E), B: batchsize, L: length, E: embedding dim
            a list of integers(actual sequence length)
            a mask indicates which elements are <PAD>.
        """
        device = self.embedding.weight.device
        tokens_batch = [preprocess(s) for s in texts]
        ids_batch = [torch.tensor(self.tokens_to_ids(tokens), device=device) for tokens in tokens_batch]
        lengths = [len(ids) for ids in ids_batch]
        ids_batch = pad_sequence(ids_batch, batch_first=True, padding_value=0)
        pad_mask = (ids_batch == 0)
        return self.embedding(ids_batch), lengths, pad_mask

    def tokens_to_ids(self, tokens):
        ids = [self.stoi[token] if token in self.stoi else 1 for token in tokens]
        return ids

    def ids_to_tokens(self, ids):
        tokens = [self.itos[id] for id in ids]
        return tokens

