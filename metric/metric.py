# -*- coding: utf-8 -*-
"""Metrics Class"""
from typing import List

from pathlib import Path

import numpy as np
import pandas as pd

import torch

from metric.cmx import plot_cmx

pd.set_option('display.unicode.east_asian_width', True)

class Metric:
    def __init__(self, n_classes: int, classes: List, metric_dir: str, eps=1e-9):
        self.n_classes = n_classes
        self.classes = classes
        self.metric_dir = Path(metric_dir)
        self.eps = eps

        self.loss_list = []
        self.cmx = torch.zeros(self.n_classes, self.n_classes, dtype=torch.int64)

    def update(self, preds, targets, loss):

        stacked = torch.stack((targets, preds), dim=1)
        for p in stacked:
            tl, pl = p.tolist()
            self.cmx[tl, pl] = self.cmx[tl, pl] + 1

        self.loss_list.append(loss)
        
    def result(self, epoch: int, mode: str = 'train', eval: bool = False):
        # precision, recall, f1score
        tp = torch.diag(self.cmx).to(torch.float32)
        fp = (self.cmx.sum(axis=1) - torch.diag(self.cmx)).to(torch.float32)
        fn = (self.cmx.sum(axis=0) - torch.diag(self.cmx)).to(torch.float32)

        self.precision = tp / (tp + fp + self.eps)
        self.recall = tp / (tp + fn + self.eps)
        self.f1score = tp / (tp + 0.5 * (fp + fn) + self.eps) # micro f1score

        # accuracy
        self.acc = (100.0 * tp / torch.sum(self.cmx)).item()

        # loss
        self.loss = mean(self.loss_list).item()

        # logging
        self._logging(epoch, mode)

        # save csv
        self._save_csv(epoch, mode)

        # plot confusion matrix
        if eval:
            cmx_path = self.metric_dir / 'test_cmx.png'
            plot_cmx(self.cmx.clone().numpy(), self.classes, self.cmx_path)

    def reset_states(self):
        self.loss_list = []
        self.cmx = torch.zeros(self.n_classes, self.n_classes, dtype=torch.int64)

    def _logging(self, epoch, mode):
        pass
    #     logger.info(f'{mode} metrics...')
    #     logger.info(f'loss:         {self.loss}')
    #     logger.info(f'accuracy:     {self.accuracy}')

    #     df = pd.DataFrame(index=self.classes)
    #     df['precision'] = self.precision.tolist()
    #     df['recall'] = self.recall.tolist()
    #     df['f1score'] = self.f1score.tolist()

    #     logger.info(f'\nmetrics values per classes: \n{df}\n')

    #     logger.info(f'precision:    {self.precision.mean()}')
    #     logger.info(f'recall:       {self.recall.mean()}')
    #     logger.info(f'mean_f1score: {self.f1score.mean()}\n') # micro mean f1score

            # plot confusion matrix
            # if eval:
            #     cmx_path = Path(self.metric_dir) / 'test_cmx.png'
            #     plot_cmx(self.cmx.clone().numpy(), self.classes, self.cmx_path)

    def _save_csv(self, epoch: int, mode: str):
        """Save results to csv"""
        csv_path = self.metrics_dir / f'{mode}_metrics.csv'

        if not csv_path.exists():
            with open(csv_path, 'w') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(['epoch', f'{mode} loss', f'{mode} accuracy',
                                    f'{mode} precision', f'{mode} recall', f'{mode} micro f1score'])

        with open(csv_path, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, self.loss, self.accuracy, 
                                self.precision.mean().item(), self.recall.mean().item(), self.f1score.mean().item()])