from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import torch
from . import *

class Visualization:
    
    def _vis_augmentation(cls, dataset):
        os.makedirs(cfg['Debug']['transforms'], exist_ok=True)
        range_idxs = list(range(0, 30))
        aug = TransformCRNN()
        for idx in range_idxs:
            img, __, __ = dataset[idx]
            img = aug.augment(img)
            cv2.imwrite(os.path.join(cfg['Debug']['transform'], f'{idx}.png'), img)
    
    def _vis_model(cls, model, size_int):
        from torchview import draw_graph
        x = torch.randn(size=size_int).to(model.device)
        draw_graph(model, input_size=x.unsqueeze(0).shape,
                   expand_nested=True,
                   save_graph=True,
                   directory=cfg['Debug']['model'],
                   graph_name='crnn')

    def _vis_error_preds(cls):
        pass