from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn

class CTCLoss(nn.Module):
    def __init__(self) -> None:
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')

    def forward(self, log_probs, targets, pred_lenghts, target_lenghts):
        loss = self.loss_func(log_probs, targets, pred_lenghts, target_lenghts)
        return loss.mean()
    