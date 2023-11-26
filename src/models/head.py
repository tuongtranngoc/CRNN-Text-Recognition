from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn

class HeadCRNN(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(HeadCRNN, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels
    
    def forward(self, x):
        x = self.fc(x)
        return x