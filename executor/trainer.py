# -*- coding: utf-8 -*-
"""Trainer Class"""
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from utils.logger import get_logger

LOG = get_logger(__name__)

class Trainer:

    def __init__(self, **kwargs):
        self.device = kwargs['device']
        self.model = kwargs['model']
        self.trainloader, self.testloader = kwargs['dataloaders']
        self.epochs = kwargs['epochs']
        self.optimizer = kwargs['optimizer']
        self.criterion = kwargs['criterion']
        self.metric = kwargs['metrics']
        self.save_ckpt_interval = kwargs['save_ckpt_interval']
        self.ckpt_dir = kwargs['ckpt_dir']
        self.writer = SummaryWriter(str(kwargs['summary_dir']))

    def train(self):
        best_acc = 0.0

        for epoch in range(self.epochs):
            LOG.info(f'\n==================== Epoch: {epoch} ====================')
            LOG.info('\n Train:')
            self.model.train()

            with tqdm(self.trainloader, ncols=100) as pbar:
                for idx, (inputs, targets) in enumerate(pbar):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)

                    loss = self.criterion(outputs, targets)

                    loss.backward()

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    preds = outputs.argmax(axis=1)
                    self.metric.update(preds=preds.cpu().detach().clone(),
                                        targets=targets.cpu().detach().clone(),
                                        loss=loss.item())

                    pbar.set_description(f'train epoch:{epoch}')

            self.metric.result(epoch, mode='train')
            self._write_summary(epoch, mode='train')
            self.metric.reset_states()

            # eval
            eval_acc = self.eval(epoch)

            # save ckpt
            if epoch != 0 and epoch % self.save_ckpt_interval == 0:
                LOG.info(' Saving Checkpoint...')
                self._save_ckpt(epoch)

            # save best ckpt
            if eval_acc > best_acc:
                best_acc = eval_acc
                self._save_ckpt(epoch, mode='best')

    def eval(self, epoch: int = 0):
        self.model.eval()

        with torch.no_grad():
            with tqdm(self.testloader, ncols=100) as pbar:
                for idx, (inputs, targets) in enumerate(pbar):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)

                    loss = self.criterion(outputs, targets)
                    self.optimizer.zero_grad()

                    preds = outputs.argmax(axis=1)
                    self.metric.update(preds=preds.cpu().detach().clone(),
                                        targets=targets.cpu().detach().clone(),
                                        loss=loss.item())

                    pbar.set_description(f'eval epoch: {epoch}')
    
        self.metric.result(epoch, mode='eval')
        self._write_summary(epoch, mode='eval')

        eval_acc = self.metric.acc
        self.metric.reset_states()

        return eval_acc

    def _write_summary(self, epoch: int, mode: str):
        # Change mode from 'eval' to 'val' to change the display order from left to right to train and eval.
        mode = 'val' if mode == 'eval' else mode

        self.writer.add_scalar(f'loss/{mode}', self.metric.loss, epoch)
        self.writer.add_scalar(f'accuracy/{mode}', self.metric.acc, epoch)
        self.writer.add_scalar(f'mean_f1score/{mode}', self.metric.f1score.mean(), epoch)
        self.writer.add_scalar(f'precision/{mode}', self.metric.precision.mean(), epoch)
        self.writer.add_scalar(f'recall/{mode}', self.metric.recall.mean(), epoch)

    def _save_ckpt(self, epoch, mode=None, zfill=4):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
    
        if mode == 'best':
            ckpt_path = self.ckpt_dir / 'best_acc_ckpt.pth'
        else:
            ckpt_path = self.ckpt_dir / f'epoch{str(epoch).zfill(zfill)}_ckpt.pth'

        torch.save({
            'epoch': epoch,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, ckpt_path)