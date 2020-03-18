import torch
from torch import nn

class xEntCriterion(nn.Module):
    def __init__(self):
        super(xEntCriterion, self).__init__()
        self.xeloss = nn.CrossEntropyLoss()

    def forward(self, score_pos, score_neg):
        logits = torch.cat([score_pos, score_neg], 1)
        targets = torch.zeros(logits.shape[0]).long().to(logits.device)
        loss = self.xeloss(logits, targets)
        return loss, {}