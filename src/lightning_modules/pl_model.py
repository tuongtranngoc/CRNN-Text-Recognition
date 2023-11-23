from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.models.crnn import CRNN
from src.models.ctc_decode import *
from src.utils.losses import CTCLoss

from . import *


class LitCRNN(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.num_classes = cfg['Global']['num_classes']
        self.model = CRNN(self.num_classes)
        self.loss_func = CTCLoss()
        self.best_acc = 0.0

    def on_train_start(self):
        pass

    def training_step(self, batch, batch_idx):
        train_metrics = BatchMeter()
        images, labels, labels_len = batch
        bz = images.size(0)
        outs = self.model(images)
        log_probs = F.log_softmax(outs, dim=2)
        labels_len = torch.flatten(labels_len)
        images_len = torch.tensor([outs.size(0)] * bz, dtype=torch.long)
        loss = self.loss_func(log_probs, labels, images_len, labels_len)
        train_metrics.update(loss)
        self.log("loss", train_metrics.get_value("mean"), prog_bar=False)
        optim = self.optimizers
        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()
        self.lr_schedulers().step()


    def validation_step(self, batch, batch_idx):
        images, labels, labels_len = batch
        bz = images.size(0)
        outs = self.model(images)
        log_probs = F.log_softmax(outs, dim=2)
        labels_len = torch.flatten(labels_len)
        images_len = torch.tensor([outs.size(0)] * bz, dtype=torch.long)

        log_probs = log_probs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy().tolist()
        
        preds = best_path_decode(log_probs)
        acc = compute_acc(preds, labels, labels_len)
        return acc
        
    def on_validation_epoch_end(self, val_step_outs):
        val_metrics = BatchMeter()
        for acc in val_step_outs:
            val_metrics.update(acc)
        current_acc = val_metrics.get_value("mean")
        self.log("acc", current_acc, prog_bar=False)
        if current_acc > self.best_acc:
            self.best_acc = current_acc
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        return [optimizer]

    def on_train_end(self):
        self.log(f"Best acc: {self.best_acc :.3f}")


def compute_acc(preds, labels, labels_len):
    correct_num = 0
    all_num = 0
    new_labels = []
    i = 0
    for char_len in labels_len:
        new_labels.append(labels[i: i+char_len])
        i += char_len
    for (pred), (target) in zip(preds, new_labels):
        pred = ''.join([id2char[int(c)] for c in pred])
        target = ''.join([id2char[int(c)] for c in target])
        pred = pred.replace(" ", "")
        target = target.replace(" ", "")
        if pred == target:
            correct_num += 1
        all_num += 1
    correct_num += correct_num
    all_num += all_num

    return correct_num/all_num

def id2char():
    dict_id2char = {}
    with open(cfg['Global']['character_dict_path'],'r', encoding='utf-8') as f_dict:
        char_list = f_dict.readlines()
        for i, char in enumerate(char_list):
            char = char.strip('\n')
            if char not in dict_id2char:
                dict_id2char[i + 1] = char
    f_dict.close()
    return dict_id2char