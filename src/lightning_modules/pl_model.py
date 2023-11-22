from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import pytorch_lightning as pl


class LitCRNN(pl.LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    
    def on_train_start(self):
        pass

    def training_step(self, batch, batch_idx):
        images, labels, labels_len = batch
        bz = images.size(0)


    def validation_step(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        pass

    def on_train_end(self):
        pass

    def forward(self):
        pass

