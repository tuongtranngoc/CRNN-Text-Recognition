from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import numpy as np
from typing import Tuple, List

import torch
import torchvision
    
from . import *


class DataUtils:
    @classmethod
    def to_device(cls, data):
        if isinstance(data, torch.Tensor):
            return data.to(cfg['Global']['device'])
        elif isinstance(data, Tuple) or isinstance(data, List):
            for i, d in enumerate(data):
                if isinstance(d, torch.Tensor):
                    data[i] = d.to(cfg['Global']['device'])
                else:
                    Exception(f"{d} in {data} is not a tensor type")
            return data
        elif isinstance(data, torch.nn.Module):
            return data.to(cfg['Global']['device'])
        else:
            Exception(f"{data} is not a/tuple/list of tensor type")