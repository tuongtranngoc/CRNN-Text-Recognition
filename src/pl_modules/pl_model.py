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

logger = Logger.get_logger("TRAINING")

class LitCRNN(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.char2id, self.id2char = map_char2id()
        self.num_classes = len(self.id2char)
        self.model = CRNN(self.num_classes)
        self.loss_func = CTCLoss()
        self.best_acc = 0.0
        self.val_metrics = BatchMeter()
        self.train_metrics = BatchMeter()
        self.automatic_optimization = False

    def on_train_start(self):
        pass

    def training_step(self, batch, batch_idx):
        images, labels, labels_len = batch
        bz = images.size(0)
        outs = self.model(images)
        log_probs = F.log_softmax(outs, dim=2)
        labels_len = torch.flatten(labels_len)
        images_len = torch.tensor([outs.size(0)] * bz, dtype=torch.long)
        loss = self.loss_func(log_probs, labels, images_len, labels_len)
        self.train_metrics.update(loss)
        self.log("Loss", self.train_metrics.get_value("mean"), prog_bar=False)
        optim = self.optimizers()
        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()

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
        acc = compute_acc(preds, labels, labels_len, self.id2char)
        self.val_metrics.update(acc)
        return acc
        
    def on_validation_epoch_end(self):
        current_acc = self.val_metrics.get_value("mean")
        self.log("Acc", current_acc, prog_bar=False)
        logger.info(f"Acc: {current_acc :.3f}")
        if current_acc > self.best_acc:
            self.best_acc = current_acc
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg['Optimizer']['lr'], amsgrad=True)
        return [optimizer]

    def on_train_end(self):
        logger.info(f"Loss: {self.train_metrics.get_value('mean')}")
        self.log(f"Best acc: {self.best_acc :.3f}")
