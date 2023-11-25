from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn

class CTCLoss(nn.Module):
    def __init__(self) -> None:
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none', zero_infinity=True)
    
    def forward(self, log_probs, targets, pred_lenghts, target_lenghts):
        """ log_probs: (T, N, C)
            targets: sum(target_lengths)
            pred_lenghts: N - batch size
            target_lenghts: N - batch size
        """
        loss = self.loss_func(log_probs, targets, pred_lenghts, target_lenghts)
        return loss.mean()
    

class CTCFacalLoss(nn.Module):
    def __init__(self) -> None:
        super(CTCFacalLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none', zero_infinity=True)
    
    def forward(self, log_probs, targets, pred_lenghts, target_lenghts):
        """ log_probs: (T, N, C)
            targets: sum(target_lengths)
            pred_lenghts: N - batch size
            target_lenghts: N - batch size
        """
        loss = self.loss_func(log_probs, targets, pred_lenghts, target_lenghts)
        weight = torch.exp(-loss)
        weight = torch.subtract(torch.tensor([1.0], device=log_probs.device), weight)
        weight = torch.square(weight)
        loss = torch.multiply(loss, weight)
        return loss.mean()
    