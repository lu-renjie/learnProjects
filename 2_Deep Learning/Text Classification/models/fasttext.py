import torch
import torch.nn as nn

from .lstm import Embedding


class FastText(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding = Embedding(
            input_dim,
            freeze=True,
            cache_dir='/Users/lurenjie/Documents/4_pretrained/glove.6B'
        )
        self.classifier = nn.Linear(input_dim, 2)

    def forward(self, texts):
        sequence, lengths, pad_mask = self.embedding(texts)
        feature = sequence.mean(dim=1)
        logits = self.classifier(feature)
        return logits
