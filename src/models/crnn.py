from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn

from .backbone import mobilenetv3_extractor
from .neck import NeckCRNN
from .head import HeadCRNN


class CRNN(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(CRNN, self).__init__()
        self.feat_extract = mobilenetv3_extractor()
        self.neck = NeckCRNN(576)
        self.head = HeadCRNN(192, 187)

    def forward(self, x):
        x = self.feat_extract(x)
        x = self.neck(x)
        x = self.head(x)
        return x