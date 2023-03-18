import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class RDropLoss(nn.Module):
    def __init__(self, weight, alpha=5):
        super().__init__()
        w = torch.FloatTensor(weight)
        self.ce = nn.CrossEntropyLoss(weight=w, reduction='none')
        self.kld = nn.KLDivLoss(reduction='none')
        self.alpha = alpha

    def forward(self, logits1, logits2, gt):
        ce_loss = (self.ce(logits1, gt) + self.ce(logits2, gt)) / 2
        kl_loss1 = self.kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).sum(-1)
        kl_loss2 = self.kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).sum(-1)
        kl_loss = (kl_loss1 + kl_loss2) / 2
        loss = ce_loss + self.alpha * kl_loss

        loss = loss.mean(-1)
        print(loss.shape)
        return loss


rdl = RDropLoss(weight=[1, 1], alpha=5)

l1 = torch.randn((2, 2))
l2 = torch.randn((2, 2))
gt = torch.tensor([1, 0])
print(l1.shape)
print(l2.shape)
print(gt.shape)

loss = rdl(l1, l2, gt)
print(loss)
