# -*- coding: utf-8 -*-
"""Setup Device"""
import torch

from utils.logger import get_logger

LOG = get_logger(__name__)

def setup_device(n_gpus: int):
    """ Setup GPU or CPU """
    if n_gpus > 0 and torch.cuda.is_available():
        LOG.info('\n CUDA is available! using GPU...')
        return torch.device('cuda')
    else:
        LOG.info('\n Using CPU...')
        return torch.device('cpu')

def data_parallel(model: object, n_gpus: int):
    """DataParallel"""
    LOG.info('\n Multiple GPUs. Data Parallel...')
    return torch.nn.DataParallel(model)