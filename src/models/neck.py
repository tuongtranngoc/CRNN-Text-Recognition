from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn


class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs) -> None:
        super().__init__()
        self.out_channels = in_channels
    
    def forward(self, x):
        __, __, H, ___ = x.shape
        # Mentioned in paper: It is unconstrained to the lengths of sequence-like objects,
        # requiring only height normalization in both training and testing phases.
        # assert H == 1
        x = x.squeeze(dim=2)
        x = x.permute(2, 0, 1) # NCW -> WNC before inputing to LSTM (input shape: WNC)
        return x
    

class EncoderRNN(nn.Module):
    def __init__(self, in_channels, hidden_size) -> None:
        super(EncoderRNN, self).__init__()
        self.out_channels = hidden_size * 2
        self.bilstm = nn.LSTM(in_channels, hidden_size, bidirectional=True, num_layers=2)

    def forward(self, x):
        x, __ = self.bilstm(x)
        return x


class NeckCRNN(nn.Module):
    def __init__(self, in_channels, hidden_size=256, **kwargs) -> None:
        super(NeckCRNN, self).__init__()
        self.im2seq = Im2Seq(in_channels)
        self.encoder = EncoderRNN(self.im2seq.out_channels, hidden_size)
        self.out_channels = self.encoder.out_channels
    
    def forward(self, x):
        x = self.im2seq(x)
        x = self.encoder(x)
        return x