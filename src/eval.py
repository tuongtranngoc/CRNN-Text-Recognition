from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config as cfg

import os
import argparse
from src.models.ctc_decode import *
from src.utils.logger import Logger
from src.utils.losses import CTCLoss
from src.utils.metrics import BatchMeter
from src.utils.torch_utils import DataUtils
from src.data.dataset_lmdb import lmdb_collate_fn

logger = Logger.get_logger("EVALUATION")


class Evaluation(object):
    def __init__(self, valid_dataset, model) -> None:
        self.args = cli()
        self.model = model
        self.valid_dataset = valid_dataset
        self.loss_func = CTCLoss()
        __, self.id2char = valid_dataset.map_char2id()
        self.data_loader = DataLoader(self.valid_dataset,
                                      batch_size=self.args.batch_size,
                                      shuffle=self.args.shuffle,
                                      num_workers=self.args.num_workers,
                                      pin_memory=self.args.pin_memory,
                                      collate_fn=lmdb_collate_fn)
        
    def evaluate(self):
        metrics = {
            'eval_loss': BatchMeter(),
            'eval_acc': BatchMeter()
        }
        self.model.eval()
        for i, (images, labels, labels_len) in enumerate(self.data_loader):
            with torch.no_grad():
                bz = images.size(0)
                images = DataUtils.to_device(images)
                labels = DataUtils.to_device(labels)

                out = self.model(images)
                log_probs = F.log_softmax(out, dim=2)
                labels_len = torch.flatten(labels_len)
                images_len = torch.tensor([out.size(0)] * bz, dtype=torch.long)
                
                loss = self.loss_func(log_probs, labels, images_len, labels_len)

                log_probs = log_probs.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy().tolist()
                
                preds = best_path_decode(log_probs)
                acc = self.compute_acc(preds, labels, labels_len)

                metrics['eval_loss'].update(loss)
                metrics['eval_acc'].update(acc)
        
        logger.info(f'loss: {metrics["eval_loss"].get_value("mean"): .4f}, acc: {metrics["eval_acc"].get_value("mean"): .4f}')
        return metrics
    
    def compute_acc(self, preds, labels, labels_len):
        correct_num = 0
        all_num = 0
        new_labels = []
        i = 0
        for char_len in labels_len:
            new_labels.append(labels[i: i+char_len])
            i += char_len
        for (pred), (target) in zip(preds, new_labels):
            pred = ''.join([self.id2char[int(c)] for c in pred])
            target = ''.join([self.id2char[int(c)] for c in target])
            pred = pred.replace(" ", "")
            target = target.replace(" ", "")
            if pred == target:
                correct_num += 1
            all_num += 1
        correct_num += correct_num
        all_num += all_num

        return correct_num/all_num
    
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=cfg['Eval']['loader']['batch_size'])
    parser.add_argument("--shuffle", default=cfg['Eval']['loader']['shuffle'])
    parser.add_argument("--num_workers", default=cfg['Eval']['loader']['num_workers'])
    parser.add_argument("--pin_memory", default=cfg['Eval']['loader']['use_shared_memory'])
    parser.add_argument("--device", default=cfg['Global']['device'])
    parser.add_argument("--lr", default=cfg['Optimizer']['lr'])
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    eva = Evaluation(args)
    eva.evaluate()