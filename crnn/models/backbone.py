from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torchvision
import torch.nn as nn

class FeatureSequenceExtractor():
    def __init__(self) -> None:
        pass
        
    def mobilenetv3_extractor(self):
        backbone = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        features = backbone.features[:-4]
        features.extend([
            nn.Conv2d(in_channels=48, out_channels=144, kernel_size=1),
            nn.Conv2d(in_channels=144, out_channels=288, kernel_size=1),
            nn.MaxPool2d(kernel_size=2)
        ])
        return nn.Sequential(*features)
