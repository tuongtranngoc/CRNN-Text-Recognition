from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from torch.utils.data import Dataset
import numpy as np
import lmdb
import cv2
import os

from . import *

import torch
from tqdm import tqdm

logger = Logger.get_logger("DATASET")


class LMDBDataSet(Dataset):
    def __init__(self, mode):
        super(LMDBDataSet, self).__init__()
        self.char2id, self.id2char = self.map_char2id()
        self.num_classes = len(self.char2id) + 1
        logger.info(f"Preparing dataset for {mode}")
        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(cfg[mode]['dataset']['data_dir'])
        self.data_idx_order_list = self.dataset_traversal()
        self.augment = TransformCRNN()
        self.is_aug = cfg[mode]['dataset']['transforms']['augmentation']

    def map_char2id(self):
        dict_char2id = {}
        dict_id2char = {}
        with open(cfg['Global']['character_dict_path'],'r', encoding='utf-8') as f_dict:
            char_list = f_dict.readlines()
            for i, char in enumerate(char_list):
                char = char.strip('\n')
                if char not in dict_char2id:
                    dict_char2id[char] = i + 1
                    dict_id2char[i + 1] = char
        f_dict.close()
        return dict_char2id, dict_id2char

    def load_hierarchical_lmdb_dataset(self, data_dir):
        lmdb_sets = {}
        dataset_idx = 0
        for (dirpath, dirnames, filenames) in os.walk(data_dir + '/'):
            if not dirnames:
                env = lmdb.open(
                    dirpath,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)
                txn = env.begin(write=False)
                num_samples = int(txn.get('num-samples'.encode()))
                lmdb_sets[dataset_idx] = {"dirpath":dirpath, "env":env, \
                    "txn":txn, "num_samples":num_samples}
                dataset_idx += 1
        return lmdb_sets

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['num_samples']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]['num_samples']
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] \
                = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            beg_idx = beg_idx + tmp_sample_num
        return data_idx_order_list

    def get_img_data(self, value):
        if not value:
            return None
        imgdata = np.frombuffer(value, dtype='uint8')
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)
        if imgori is None:
            return None
        return imgori

    def get_lmdb_sample_info(self, txn, index):
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode('utf-8')
        label_id = [self.char2id[c.lower()] for c in label if c.lower() in self.char2id]
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        image = self.get_img_data(imgbuf)
        if self.is_aug:
            image = self.augment.augment(image)
        image = self.augment.transform(image)
        return image, label_id

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'],
                                                file_idx)
        if sample_info is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        img, label = sample_info
        label = torch.tensor(label, dtype=torch.long)
        label_len = torch.tensor([len(label)], dtype=torch.long)
        return img, label, label_len

    def __len__(self):
        return self.data_idx_order_list.shape[0]


def lmdb_collate_fn(batch):
    images, labels, label_len = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.cat(labels, dim=0)
    label_len = torch.cat(label_len, dim=0)
    return images, labels, label_len