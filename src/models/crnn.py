from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn

from .backbone import features_sequence_extractor
from .neck import NeckCRNN
from .head import HeadCRNN


class CRNN(nn.Module):
    def __init__(self, num_classes=37,**kwargs) -> None:
        super(CRNN, self).__init__()
        self.feat_extract = features_sequence_extractor()
        self.neck = NeckCRNN(512)
        self.head = HeadCRNN(512, num_classes)
    
    def forward(self, x):
        x = self.feat_extract(x)
        x = self.neck(x)
        sq_len, bz, in_size = x.size()
        sq_len2 = x.view(sq_len*bz, in_size)
        x = self.head(sq_len2)
        x = x.view(sq_len, bz, -1)
        return x