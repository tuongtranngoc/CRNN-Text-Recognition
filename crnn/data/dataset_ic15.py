from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import torch
import torch.nn as nn

import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from . import *

class Icdar15Dataset(nn.Module):
    def __init__(self):
        self.dataset = self.load_dataset()

    def load_dataset(self, rec_type='Train'):
        dataset = []
        label_file_list = cfg[rec_type]['dataset']['label_file_list']
        for label_file in label_file_list:
            with open(label_file, 'r', encoding='utf-8') as f_label:
                label_lines = f_label.readlines()
                for line in label_lines:
                    pth_img, text = line.strip().split('\t')
                    pth_img = os.path.basename(pth_img)
                    pth_img = os.path.join(cfg[rec_type]['dataset']['data_dir'])
                    dataset.append([pth_img, text])

    def __transform(self):        
        pass
    
    def get_image_label(self, img_pth, labels, is_aug):
        image = cv2.imread(img_pth)
        image = image[..., ::-1]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        pass


if __name__ == "__main__":
    icdar15 = Icdar15Dataset()
    icdar15.load_dataset()
