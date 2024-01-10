from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    def forward(self, log_probs, targets, pred_lengths, target_lengths):
        """ log_probs: (T, N, C)
            targets: sum(target_lengths)
            pred_lenghts: N - batch size
            target_lenghts: N - batch size
        """
        loss = self.loss_func(log_probs, targets, pred_lengths, target_lengths)
        weight = torch.exp(-loss)
        weight = torch.subtract(torch.tensor([1.0], device=log_probs.device), weight)
        weight = torch.square(weight)
        loss = torch.multiply(loss, weight)
        return loss.mean()
    

class CTCACELoss(nn.Module):
    def __init__(self) -> None:
        super(CTCACELoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(
            weight=None,
            ignore_index=0,
            reduction='none',
            soft_label=True,
            axis=-1)

    def forward(self, log_probs, targets, pred_lengths, target_lengths):
        log_probs = log_probs.permute((1, 0, 2))
        log_probs = log_probs[-1]
        B, N = log_probs.shape
        div = torch.tensor([N], dtype=torch.float32)
        predicts = F.softmax(log_probs, dim=-1)
        aggregation_preds = torch.sum(predicts, dim=1)
        aggregation_preds = torch.divide(aggregation_preds, div)
        
    