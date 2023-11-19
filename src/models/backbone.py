from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torchvision
import torch.nn as nn


def mobilenetv3_extractor():
    backbone = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    features = backbone.features[:-4]
    features.extend([
        nn.Conv2d(in_channels=48, out_channels=288, kernel_size=1),
        nn.Conv2d(in_channels=288, out_channels=576, kernel_size=1),
        nn.MaxPool2d(kernel_size=2)
    ])
    return nn.Sequential(*features)
