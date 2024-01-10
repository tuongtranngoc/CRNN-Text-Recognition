from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.crnn import CRNN
from src.utils.logger import Logger
from src.utils.metrics import BatchMeter
from src.utils.torch_utils import DataUtils
from src.utils.tensorboard import Tensorboard
from src.utils.losses import CTCLoss, CTCFacalLoss, CTCACELoss
from src.data.dataset_lmdb import LMDBDataSet, lmdb_collate_fn

from . import config as cfg
from src.eval import Evaluation

import os
import argparse

logger = Logger.get_logger("TRAINING")


class Trainer(object):
    def __init__(self, args) -> None:
        self.args = args
        self.start_epoch = 1
        self.best_acc = 0.0
        self.create_data_loader()
        self.create_model()
        self.eval = Evaluation(self.valid_dataset, self.model)
    
    def create_data_loader(self):
        self.train_dataset = LMDBDataSet(mode='Train')
        self.valid_dataset = LMDBDataSet(mode='Eval')
        self.train_loader = DataLoader(self.train_dataset, 
                                       batch_size=self.args.batch_size, 
                                       shuffle=self.args.shuffle,
                                       num_workers=self.args.num_workers,
                                       pin_memory=self.args.pin_memory,
                                       collate_fn=lmdb_collate_fn) # Add custom collate_fn to dataloader
    
    def create_model(self):
        self.model = CRNN(num_classes=self.train_dataset.num_classes).to(self.args.device)
        self.loss_func = CTCACELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)

        if self.args.resume:
            logger.info("Resuming training ...")
            last_ckpt = self.args.last_ckpt_pth
            if os.path.exists(last_ckpt):
                ckpt = torch.load(last_ckpt, map_location=self.args.device)
                self.start_epoch = self.resume_training(ckpt)
                logger.info(f"Loading checkpoint with start epoch: {self.start_epoch}, best acc: {self.best_acc}")
    
    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            mt_loss = BatchMeter()
            for i, (images, labels, labels_len) in enumerate(self.train_loader):
                self.model.train()
                bz = images.size(0)
                images = DataUtils.to_device(images)
                labels = DataUtils.to_device(labels)
                out = self.model(images)
                log_probs = F.log_softmax(out, dim=2)
                labels_len = torch.flatten(labels_len)
                images_len = torch.tensor([out.size(0)] * bz, dtype=torch.long)
                loss = self.loss_func(log_probs, labels, images_len, labels_len)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                mt_loss.update(loss.item())
                print(f"Epoch {epoch} - batch {i+1}/{len(self.train_loader)} - loss: {mt_loss.get_value()}", end='\r')

                Tensorboard.add_scalars('train_loss', epoch, loss=mt_loss.get_value("mean"))

            logger.info(f"Epoch {epoch} - loss: {mt_loss.get_value('mean'): .4f}")

            if epoch % self.args.eval_step == 0:
                metrics = self.eval.evaluate()
                Tensorboard.add_scalars('eval_loss', epoch, loss=metrics['eval_loss'].get_value("mean"))
                Tensorboard.add_scalars('eval_acc', epoch, loss=metrics['eval_acc'].get_value("mean"))
                
                # Save best checkpoint
                current_acc = metrics['eval_acc'].get_value("mean")
                if current_acc > self.best_acc:
                    self.best_acc = current_acc
                    best_ckpt_path = self.args.best_ckpt_pth
                    self.save_ckpt(best_ckpt_path, self.best_acc, epoch)
            
            # Save last checkpoint
            last_ckpt_path = self.args.last_ckpt_pth
            self.save_ckpt(last_ckpt_path, self.best_acc, epoch)


    def save_ckpt(self, save_path, best_acc, epoch):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ckpt_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": best_acc,
            "epoch": epoch
        }
        logger.info(f"Saving checkpoint to {save_path}")
        torch.save(ckpt_dict, save_path)

    def resume_training(self, ckpt):
        self.best_acc = ckpt['best_acc']
        start_epoch = ckpt['epoch'] + 1
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.model.load_state_dict(ckpt['model'])

        return start_epoch
    

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=cfg['Optimizer']['lr'])
    parser.add_argument("--device", default=cfg['Global']['device'])
    parser.add_argument("--resume", default=cfg['Global']['resume_training'])
    parser.add_argument("--epochs", default=cfg['Train']['loader']['epochs'])
    parser.add_argument("--shuffle", default=cfg['Train']['loader']['shuffle'])
    parser.add_argument("--eval_step", default=cfg['Train']['loader']['eval_step'])
    parser.add_argument("--batch_size", default=cfg['Train']['loader']['batch_size'])
    parser.add_argument("--num_workers", default=cfg['Train']['loader']['num_workers'])
    parser.add_argument("--pin_memory", default=cfg['Train']['loader']['use_shared_memory'])
    parser.add_argument("--last_ckpt_pth", default=cfg['Train']['checkpoint']['last_path'])
    parser.add_argument("--best_ckpt_pth", default=cfg['Train']['checkpoint']['best_path'])
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    trainer = Trainer(args)
    trainer.train()