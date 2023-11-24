from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torchvision
import torch.nn as nn


def features_sequence_extractor():
    backbone = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),  # image size: 16 * 64
            
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),  # image size: 8 * 32

            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # image size: 4 x 16

            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # image size: 2 x 16

            nn.Conv2d(512, 512, (2, 2), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  # image size: 1 x 16
        )
    return backbone