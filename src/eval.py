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
from tqdm import tqdm

from src.models.crnn import CRNN
from src.models.ctc_decode import *
from src.utils.logger import Logger
from src.utils.losses import CTCLoss
from src.utils.torch_utils import DataUtils
from src.data.dataset_lmdb import lmdb_collate_fn, LMDBDataSet
from src.utils.metrics import BatchMeter, compute_acc, map_char2id

logger = Logger.get_logger("EVALUATION")


class Evaluation(object):
    def __init__(self, valid_dataset, model) -> None:
        self.args = cli()
        self.model = model
        self.valid_dataset = valid_dataset
        self.loss_func = CTCLoss()
        __, self.id2char = map_char2id()
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
        for images, labels, labels_len in tqdm(self.data_loader):
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
                acc = compute_acc(preds, labels, labels_len, self.id2char)

                metrics['eval_loss'].update(loss)
                metrics['eval_acc'].update(acc)
        
        logger.info(f'loss: {metrics["eval_loss"].get_value("mean"): .4f}, acc: {metrics["eval_acc"].get_value("mean"): .4f}')
        return metrics
    
    
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=cfg['Eval']['loader']['batch_size'])
    parser.add_argument("--shuffle", type=bool, default=cfg['Eval']['loader']['shuffle'])
    parser.add_argument("--num_workers", type=int,  default=cfg['Eval']['loader']['num_workers'])
    parser.add_argument("--pin_memory", type=bool, default=cfg['Eval']['loader']['use_shared_memory'])
    parser.add_argument("--device", type=str, default=cfg['Global']['device'])
    parser.add_argument("--lr", type=float, default=cfg['Optimizer']['lr'])
    parser.add_argument("--model_path", type=str, default=cfg['Train']['checkpoint']['best_path'])
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    valid_dataset = LMDBDataSet("Eval")
    model = CRNN(valid_dataset.num_classes).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device)['model'])
    eva = Evaluation(valid_dataset, model)
    eva.evaluate()