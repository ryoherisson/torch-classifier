# -*- coding: utf-8 -*-
"""Make Criterion"""
import torch.nn as nn

def make_criterion(criterion_cfg: object):
    if criterion_cfg.type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError('This loss function is not supported.')