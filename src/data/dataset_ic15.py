from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from . import *

from tqdm import tqdm

logger = Logger.get_logger("DATASET")


class Icdar15Dataset(Dataset):
    def __init__(self, mode='Train'):
        self.mode = mode
        self.char2id, self.id2char = self.map_char2id()
        self.augment = TransformCRNN()
        self.dataset = self.load_dataset()

    def map_char2id(self):
        dict_char2id = {}
        dict_id2char = {}
        with open(cfg['Global']['character_dict_path'],'r', encoding='utf-8') as f_dict:
            char_list = f_dict.readlines()
            for i, char in enumerate(char_list):
                char = char.strip('\n')
                if char not in dict_char2id:
                    dict_char2id[char] = i
                    dict_id2char[i] = char
        f_dict.close()
        return dict_char2id, dict_id2char


    def load_dataset(self):
        dataset = []
        label_file_list = cfg[self.mode]['dataset']['label_file_list']
        logger.info(f"Loaing dataset for {self.mode} ...")
        for label_file in label_file_list:
            with open(label_file, 'r', encoding='utf-8') as f_label:
                label_lines = f_label.readlines()
                for line in tqdm(label_lines):
                    pth_img, text = line.strip('\n').split('\t')
                    text2ids = [self.char2id[c] for c in text]
                    pth_img = os.path.basename(pth_img)
                    pth_img = os.path.join(cfg[self.mode]['dataset']['data_dir'], pth_img)
                    dataset.append([pth_img, text2ids])
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
        image, label = self.get_image_label(img_pth, label, is_aug=cfg[self.mode]['dataset']['transforms']['augmentation'])
        label = torch.tensor(label, dtype=torch.long)
        label_len = torch.tensor([len(label)], dtype=torch.long)
        return image, label, label_len


def icdar15_collate_fn(batch):
    images, labels, label_len = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.cat(labels, dim=0)
    label_len = torch.cat(label_len, dim=0)
    return images, labels, label_len


if __name__ == "__main__":
    icdar15 = Icdar15Dataset()
    icdar15.load_dataset()
