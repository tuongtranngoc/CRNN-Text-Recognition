from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
from torch.utils.data import DataLoader

from . import *

import argparse

logger = Logger.get_logger("__TRAINING__")


class Trainer(object):
    def __init__(self, args) -> None:
        self.args = args
        self.start_epoch = 1
        self.best_acc = 0.0

    def create_data_loader(self):
        self.train_dataset = Icdar15Dataset(mode='Train')
        self.valid_dataset = Icdar15Dataset(mode='Eval')
        self.train_loader = DataLoader(self.train_dataset, 
                                       batch_size=self.args.batch_size, 
                                       shuffle=self.args.shuffle,
                                       num_workers=self.args.num_workers,
                                       pin_memory=self.args.pin_memory)

    def create_model(self):
        self.model = CRNN().to(self.args.device)
        self.loss_func = CTCLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, amsgrad=True)
    

    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            mt_loss = BatchMeter()
            for bz, (images, labels) in enumerate(self.train_loader):
                self.model.train()
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                out = self.model(images)
                import pdb; pdb.set_trace()
    
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
        self.model.load_state_dict(ckpt['mode'])

        return start_epoch
    

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=config['Train']['loader']['batch_size'])
    parser.add_argument("--shuffle", default=config['Train']['loader']['shuffle'])
    parser.add_argument("--num_workers", default=config['Train']['loader']['num_workers'])
    parser.add_argument("--pin_memory", default=config['Train']['loader']['use_shared_memory'])


if __name__ == "__main__":
    args = cli()
    trainer = Trainer(args)
    trainer.train()
