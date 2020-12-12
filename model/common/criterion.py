# -*- coding: utf-8 -*-
"""Make Criterion"""
import torch.nn as nn

from utils.logger import get_logger

LOG = get_logger(__name__)

def make_criterion(criterion_cfg: object):
    if criterion_cfg.type == 'cross_entropy':
        LOG.info('\n Criterion: Cross Entropy Loss')
        return nn.CrossEntropyLoss()
    else:
        raise ValueError('This loss function is not supported.')