# -*- coding: utf-8 -*-
"""Load model checkpoint"""
from pathlib import Path

import torch

def load_ckpt(model: object, resume: str):
    """Loads Checkpoint"""
    if not Path(configs['resume']).exists():
        raise ValueError('No checkpoint found !')
    ckpt = torch.load(resume)
    return model.load_state_dict(ckpt['model_state_dict'])