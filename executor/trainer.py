# -*- coding: utf-8 -*-
"""Trainer Class"""

from pathlib import Path
from tqdm import tqdm

class Trainer:

    def __init__(self, **kwargs):
        self.device = kwargs['device']
        self.model = kwargs['model']
        self.trainloader, self.testloader = kwargs['dataloaders']
        self.epochs = kwargs['epochs']
        self.optimizer = kwargs['optimizer']
        self.criterion = kwargs['criterion']
        self.metrics = kwargs['metrics']
        self.save_ckpt_interval = kwargs['save_ckpt_interval']

    def train():
        best_acc = 0.0

        for epoch in range(self.epochs):
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

                    pred = outputs.argmax(axis=1)
                    self.metrics.update(preds=preds.cpu().detach().clone(),
                                        targets=targets.cpu().detach().clone(),
                                        loss=loss.item())

                    pbar.set_description(f'train epoch:{epoch}')

            self.metric.result()
            self.metric.reset_states()

            # eval
            self.eval(epoch)

            # save ckpt
            if epoch != 0 and epoch % self.save_ckpt_interval == 0:
                logger.info('saving checkpoint...')
                self._save_ckpt(epoch, train_loss)

            # save best ckpt
            if test_acc > best_acc:
                best_acc = test_acc
                self._save_ckpt(epoch, train_loss, mode='best')

    def eval(epoch: int = 0):
        self.model.eval()

        with torch.no_grad():
            with tqdm(self.testloader, ncols100) as pbar:
                for idx, (inputs, targets) in enumerate(pbar):

                    outputs = self.model(inputs)

                    loss = self.criterion(outputs, targets)
                    self.optimizer.zero_grad()
    
                    self.metrics.update(preds=preds.cpu().detach().clone(),
                                        targets=targets.cpu().detach().clone(),
                                        loss=loss.item())

                    pbar.set_description(f'eval epoch: {epoch}')
    
        # self.metric.result()
        # self.metric.reset_states()

    # def _write_summary(self, epoch, loss):
        # Change mode from 'test' to 'val' to change the display order from left to right to train and test.
        # mode = 'val' if mode == 'test' else mode

        # self.writer.add_scalar(f'loss/{mode}', self.loss, epoch)
        # self.writer.add_scalar(f'accuracy/{mode}', self.accuracy, epoch)
        # self.writer.add_scalar(f'mean_f1score/{mode}', self.f1score.mean(), epoch)
        # self.writer.add_scalar(f'precision/{mode}', self.precision.mean(), epoch)
        # self.writer.add_scalar(f'recall/{mode}', self.recall.mean(), epoch)

    # def _save_ckpt(self, epoch, loss, mode=None, zfill=4):
    #     if isinstance(self.model, nn.DataParallel):
    #         model = self.model.module
    
    #     if mode == 'best':
    #         ckpt_path = self.ckpt_dir / 'best_acc_ckpt.pth'
    #     else:
    #         ckpt_path = self.ckpt_dir / f'epoch{str(epoch).zfill(zfill)}_ckpt.pth'

    #     torch.save({
    #         'epoch': epoch,
    #         'network': model,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'loss': loss,
    #     }, ckpt_path)