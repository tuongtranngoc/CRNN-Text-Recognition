from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torchvision
import torch.nn as nn


def mobilenetv3_extractor():
    """Feature extraction
        Reference: https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html
    """
    backbone = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    features = backbone.features[:-4]
    features.extend([
        nn.Conv2d(in_channels=112, out_channels=288, kernel_size=1),
        nn.Conv2d(in_channels=288, out_channels=576, kernel_size=1),
        nn.MaxPool2d(kernel_size=(2, 1))
    ])
    return nn.Sequential(*features)
