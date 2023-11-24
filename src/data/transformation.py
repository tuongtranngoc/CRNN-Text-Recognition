from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from . import cfg

import cv2
import math


class TransformCRNN(object):
    def __init__(self) -> None:
        self.image_size = cfg['Train']['dataset']['transforms']['image_shape']
        # image transformation
        self.__tranform = A.Compose([
            A.Normalize(),
            ToTensorV2()])
        # image augmentation
        self.__augment = A.Compose([
            A.SafeRotate(limit=[-5, 5], p=0.3),
            A.Blur(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.MedianBlur(p=0.1, blur_limit=5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.3)
        ])

    def resize_padding(self, image):
        imgC, imgH, imgW = self.image_size
        h, w, __ = image.shape
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(image, (resized_w, imgH))
        resized_image = resized_image.astype(np.float32)
        padding_im = np.ones((imgH, imgW, imgC) ,dtype=resized_image.dtype) * 128
        padding_im[:, 0:resized_w, :] = resized_image
        padding_im = padding_im[..., ::-1]
        return padding_im
    
    def transform(self, image):
        image = self.resize_padding(image)
        transformed = self.__tranform(image=image)
        transformed_image = transformed['image']
        return transformed_image
        
    def augment(self, image):
        augmented = self.__augment(image=image)
        augmented_image = augmented['image']
        return augmented_image