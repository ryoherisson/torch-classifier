# -*- coding: utf-8 -*-
"""Make Optimizer"""
import torch.optim as optim

from utils.logger import get_logger

LOG = get_logger(__name__)

def make_optimizer(model: object, optimizer_cfg: object):
    if optimizer_cfg.type == 'sgd':
        LOG.info('\n Optimizer: SGD')
        return optim.SGD(model.parameters(), lr=optimizer_cfg.lr, momentum=optimizer_cfg.momentum, weight_decay=optimizer_cfg.decay)
    else:
        raise ValueError('This optimizer is not supported.')

