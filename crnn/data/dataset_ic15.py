from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import torch
import torch.nn as nn

from . import *

class Icdar15Dataset(nn.Module):
    def __init__(self, rec_type='Train'):
        self.dataset = self.load_dataset()
        self.augment = TransformCRNN()
        self.rec_type = rec_type

    def load_dataset(self):
        dataset = []
        label_file_list = cfg[self.rec_type]['dataset']['label_file_list']
        for label_file in label_file_list:
            with open(label_file, 'r', encoding='utf-8') as f_label:
                label_lines = f_label.readlines()
                for line in label_lines:
                    pth_img, text = line.strip().split('\t')
                    pth_img = os.path.basename(pth_img)
                    pth_img = os.path.join(cfg[self.rec_type]['dataset']['data_dir'], pth_img)
                    dataset.append([pth_img, text])
        return dataset
    
    def get_image_label(self, img_pth, label, is_aug):
        image = cv2.imread(img_pth)
        if is_aug:
            image = self.augment.augment(image)
        image = self.augment.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img_pth, label = self.dataset[index]
        image, label = self.get_image_label(img_pth, label, is_aug=cfg[self.rec_type]['dataset']['transforms']['augmentation'])
        return image, label


if __name__ == "__main__":
    icdar15 = Icdar15Dataset()
    icdar15.load_dataset()
