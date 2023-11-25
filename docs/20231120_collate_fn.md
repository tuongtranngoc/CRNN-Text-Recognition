# Everything you need to know when creating a dataloader with variable-size input using collate_fn of Pytorch

## Introduction

Default `collate_fn` collates a list of tuple into a single tuple of a batched image Tensor and a batched class label Tensor. In particular, the default `collate_fn` has the following properties:
  + It always prepends a new dimension as the batch dimension.
  + It automatically converts NumPy arrays and Python numerical values into PyTorch Tensors.
  + It preserves the data structure, e.g., if each sample is a dictionary, it outputs a dictionary with the same set of keys but batched Tensors as values (or lists if the values can not be converted into Tensors). Same for list s, tuple s, namedtuple s, etc.

<p align='center'>
    <image src='/images/default_collate_fn.jpg'>
</p>

+ Customized `collate_fn`, users may use it to achive custom batching, e.g, collating along a dimension other than the first, padding sequences of various lengths, or adding support for data types.

## How to use `collate_fn`

This example provides a simple custom collate_fn for a text recognition problem using the icdar15 dataset. Input images have a shape of (3, 32, 320), and input labels have varying lengths associated with each input image, such as (image1, 'ABCD'), (image2, 'DEFK12232'), and so on.

```python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Icdar15Dataset(Dataset):
    def __init__(self, mode='Train'):
        self.dataset = self.load_dataset()

    def map_char2id(self):
        # Do something here ...
        pass

    def load_dataset(self):
        dataset = []
        # Do something here ...
        return dataset
    
    def get_image_label(self, img_pth, label, is_aug):
        # Do something here ...
        return image, label
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img_pth, label = self.dataset[index]
        label = torch.tensor(label, dtype=torch.long)
        label_len = torch.tensor([len(label)], dtype=torch.long)
        return image, label, label_len

# Define collate_fn for icdar15 dataset
def icdar15_collate_fn(batch):
    images, labels, label_len = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.cat(labels, dim=0)
    label_len = torch.cat(label_len, dim=0)
    return images, labels, label_len

train_dataset = Icdar15Dataset(mode='Train')
train_loader = DataLoader(self.train_dataset, 
                        batch_size=self.args.batch_size, 
                        shuffle=self.args.shuffle,
                        num_workers=self.args.num_workers,
                        pin_memory=self.args.pin_memory,
                        collate_fn=icdar15_collate_fn) # Add custom collate_fn to dataloader
```

## Reference
+ https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3?u=ptrblck
+ https://pytorch.org/docs/stable/data.html#dataloader-collate-fn