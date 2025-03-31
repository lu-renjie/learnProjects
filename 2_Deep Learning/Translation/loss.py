import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, ignore_idx):
        super().__init__()
        self.ignore_idx = ignore_idx
        self.CELoss = nn.CrossEntropyLoss(ignore_index=ignore_idx)
    
    def forward(self, logits, ground_truth):
        """
        predicts: (B, L, E)
        actual_length: (B,)
        ground_truth: (B, E)
        通过将ignore_idx设置为pad_idx来忽略PAD的部分
        """
        B, L, E = logits.shape
        logits = logits.reshape(-1, E)
        ground_truth = ground_truth.reshape(-1)
        loss = self.CELoss(logits, ground_truth)
        return loss
