# -*- coding: utf-8 -*-
"""Set Device"""

import torch

def setup_device(n_gpus: int):
    """ Setup GPU or CPU """
    if configs['n_gpus'] > 0 and torch.cuda.is_available():
        # logger.info('CUDA is available! using GPU...\n')
        return torch.device('cuda')
    else:
        # logger.info('using CPU...\n')
        return torch.device('cpu')