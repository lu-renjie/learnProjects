import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .beamsearch import BeamSearch


class Encoder(nn.Module):
    def __init__(self, cfg, word_vectors, tokens):
        super().__init__()
        self.PAD = tokens['PAD']
        self.hidden_dim = cfg['hidden_dim']
        self.num_layers = cfg['num_layers']

        self.embed = nn.Embedding.from_pretrained(word_vectors, freeze=cfg['freeze'], padding_idx=self.PAD)
        self.rnn = nn.LSTM(
            input_size=word_vectors.shape[1],
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_dim))

    def forward(self, sequence, actual_length):
        """
        sequence: (B, L), L is sequence length
        actual_length: (B,)
        """
        batch_size = len(sequence)
        mask = sequence == self.PAD

        sequence = self.embed(sequence)  # (Batch size, length, embed dim)
        sequence = pack_padded_sequence(sequence, actual_length, batch_first=True, enforce_sorted=False)
        h0 = self.h0.expand(-1, batch_size, -1)
        c0 = self.c0.expand(-1, batch_size, -1)
        output, (hn, cn) = self.rnn(sequence, (h0, c0))
        output, length = pad_packed_sequence(output, batch_first=True, padding_value=self.PAD)
        return output, mask, (hn, cn)

    def forward_test(self, sequence):
        """只有一个样本
        """
        sequence = sequence.unsqueeze(0)
        sequence = self.embed(sequence)
        output, (hn, cn) = self.rnn(sequence, (self.h0, self.c0))
        return output, (hn, cn)


class Decoder(nn.Module):
    def __init__(self, cfg, word_vectors, tokens):
        super().__init__()
        self.PAD = tokens['PAD']
        self.BOS = tokens['BOS']
        self.EOS = tokens['EOS']

        self.concat = cfg['concat']
        self.beam_k = cfg['beam_k']
        self.max_length = cfg['max_length']
        self.hidden_dim = cfg['hidden_dim']
        self.num_layers = cfg['num_layers']
        self.use_attention = cfg['attention']

        self.embed = nn.Embedding.from_pretrained(word_vectors, freeze=cfg['freeze'], padding_idx=self.PAD)
        self.rnn = nn.LSTM(
            input_size=word_vectors.shape[1],
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True)

        if self.use_attention:
            self.attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=cfg['num_heads'],
                batch_first=True)
        if self.use_attention and self.concat:
            self.classifier = nn.Linear(2 * self.hidden_dim, word_vectors.shape[0])
        else:
            self.classifier = nn.Linear(self.hidden_dim, word_vectors.shape[0])

    def forward(self,
            seq1, mask,
            h0, c0,
            seq2, actual_length2):
        """teacher-forcing
        """
        B, L, E = seq1.shape

        # embed
        seq2 = self.embed(seq2)

        # forward RNN
        seq2 = pack_padded_sequence(seq2, actual_length2, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.rnn(seq2, (h0, c0))
        output, actual_length2 = pad_packed_sequence(output, batch_first=True, padding_value=self.PAD)
        output = output[:, :-1, :]  # 删掉序列最后一个元素

        # Attention
        if self.use_attention:
            # mask用来屏蔽<PAD>, True表示对应位置为<PAD>
            attn_output, attn = self.attn(key=seq1, value=seq1, query=output, key_padding_mask=mask)

            if self.concat:
                output = torch.cat([attn_output, output], dim=-1)
            else:
                output = attn_output
        
        # predict
        logits = self.classifier(output)
        return logits

    def step(self, x, h, key_value):
        output, h = self.rnn(x, h)

        if self.use_attention:
            key_value = key_value.repeat(x.size(0), 1, 1)
            attn_output, attn = self.attn(key=key_value, value=key_value, query=output)

            if self.concat:
                output = torch.cat([attn_output, output], dim=-1)  # (B, 1, 2E)
            else:
                output = attn_output  # (B, 1, E)

        logits = self.classifier(output).squeeze(1)   # (B, C)
        log_probs = torch.log_softmax(logits, dim=1)  # (B, C)
        return log_probs, h

    def forward_test(self, encoder_features, h0, c0):
        """use beam search to generate target sentence.
        """
        device = encoder_features.device
        beam_search = BeamSearch(self.beam_k, self.EOS, self.max_length)

        # 第一步根据<BOS>预测
        BOS = torch.tensor(self.BOS, device=device).reshape(1, 1)
        x = self.embed(BOS)
        log_probs, (h, c) = self.step(x, (h0, c0), encoder_features)
        selection_former, selection_next = beam_search.first(log_probs)

        # 然后预测后面的
        while beam_search.not_finished():
            selection_next = torch.tensor(selection_next, device=device).unsqueeze(1)  # (k, 1)
            x = self.embed(selection_next)  # (k, 1, E)
            h = h[:, selection_former, :]   # (num_layers, k, E)
            c = c[:, selection_former, :]   # (num_layers, k, E)
            log_probs, (h, c) = self.step(x, (h, c), encoder_features)
            selection_former, selection_next = beam_search.step(log_probs)
        return beam_search.get_best_predict()


class Seq2SeqAttention(nn.Module):
    def __init__(self, cfg,
            word_vectors1, word_vectors2,
            tokens1, tokens2):
        super().__init__()
        self.encoder = Encoder(cfg, word_vectors1, tokens1)
        self.decoder = Decoder(cfg, word_vectors2, tokens2)

    def forward(self, seq1, length1, seq2, lenght2):
        output, mask, (hn, cn) = self.encoder(seq1, length1)
        logits = self.decoder(output, mask, hn, cn, seq2, lenght2)
        return logits

    def predict(self, sequence):
        output, (hn, cn) = self.encoder.forward_test(sequence)
        predicts = self.decoder.forward_test(output, hn, cn)
        return predicts
