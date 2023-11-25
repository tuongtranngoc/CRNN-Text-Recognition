from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as cfg
from src.models.crnn import CRNN
from src.data.transformation import TransformCRNN


class Predictor:
    def __init__(self, args) -> None:
        self.args = args
        __, self.id2char = self.map_char2id()
        self.model = CRNN(len(self.id2char)+1)
        self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.args.device)['model'])
        self.model.to(self.args.device)
        self.transform = TransformCRNN()

    def preprocess(self):
        if os.path.exists(self.args.img_path):
            img = cv2.imread(self.args.img_path)
            img = self.transform.transform(img)
            img = img.unsqueeze(0)
            return img
        else:
            Exception("Not exist image path")
    
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
    
    def post_process(self, labels, blank=0):
        mapped_labels = []
        prev_label = None

        for l in labels:
            if l != prev_label:
                mapped_labels.append(l)
                prev_label = l
        mapped_labels = [l for l in mapped_labels if l != blank]
        return mapped_labels
    
    def post_decode(self, log_probs):
        log_prob = log_probs.cpu().detach().numpy()
        log_prob = log_prob.transpose((1, 0, 2))
        labels = np.argmax(log_prob[0], axis=-1)
        labels = self.post_process(labels)
        return labels
    
    def predict(self):
        with torch.no_grad():
            img = self.preprocess()
            out = self.model(img.to(self.args.device))
            log_prob = F.log_softmax(out, dim=2)
        decoded_id = self.post_decode(log_prob)
        decoded_text = ''.join([self.id2char[_id] for _id in decoded_id])
        print(decoded_text)
        return decoded_text

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default=None, help="Path to image file")
    parser.add_argument("--model_path", type=str, default=cfg['Train']['checkpoint']['best_path'], help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default='cuda', help="device inference (cuda or cpu)")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    predictor = Predictor(args)
    predictor.predict()