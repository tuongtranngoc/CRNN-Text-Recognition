from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer

from . import *
from src.models.crnn import CRNN
from src.utils.logger import Logger
from src.utils.losses import CTCLoss
from .pl_dataset import LMDBDataModule
from src.utils.metrics import BatchMeter
from src.utils.torch_utils import DataUtils
from src.utils.tensorboard import Tensorboard


def main():
    data = LMDBDataModule()
        