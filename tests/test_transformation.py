from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from src.data.transformation import TransformCRNN
from src.data.dataset_ic15 import Icdar15Dataset

import os
import cv2
import numpy as np
from . import cfg

def test():
    debug_pth = cfg['Debug']['transforms']
    os.makedirs(debug_pth, exist_ok=True)
    dataset = Icdar15Dataset()
    data = dataset.load_dataset(rec_type='Eval')
    aug = TransformCRNN()

    for i in range(len(data[:10])):
        img_pth, label = data[i]
        img = cv2.imread(img_pth)
        aug_img = aug.augment(img)

        mask_compare = np.ones((max(img.shape[0], aug_img.shape[0]), max(img.shape[1], aug_img.shape[1])*2, img.shape[2]), dtype=img.dtype)
        mask_compare[0:img.shape[0], 0:img.shape[1]] = img
        mask_compare[0:aug_img.shape[0], img.shape[1]:(aug_img.shape[1]+img.shape[1])] = aug_img

        cv2.imwrite(os.path.join(debug_pth, os.path.basename(img_pth)), mask_compare)
        


if __name__ == "__main__":
    test()
