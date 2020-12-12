# -*- coding: utf-8 -*-
"""DataLoader class"""
from typing import List
from pathlib import Path

import pandas as pd

import torch.utils.data as data

from dataloader.utils import make_data_list
from dataloader.dataset import Dataset
from dataloader.transform import DataTransform

class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(dataroot: str, labelroot: str):
        """load dataset from path

        Parameters
        ----------
        dataroot : str
            path to the image data directory e.g. './data/images/'
        labelroot : str
            path to the label csv e.g. './data/labels/train.csv/'

        Returns
        -------
        Tuple of list
            img_list: e.g. ['./data/images/car1.png', './data/images/dog4.png', ...]
            lbl_list: e.g. [3, 5, ...]
        """
        img_list, lbl_list = make_data_list(dataroot, labelroot)
        return (img_list, lbl_list)

    @staticmethod
    def preprocess_data(data_config: object, img_list: List, lbl_list: List, batch_size: int, mode: str):
        """Preprocess dataset

        Parameters
        ----------
        data_config : object
            data configuration
        img_list : List
            a list of image paths
        lbl_list : List
            a list of labels
        batch_size : int
            batch_size
        mode : str
            'train' or 'eval'

        Returns
        -------
        Object : 
            DataLoader instance

        Raises
        ------
        ValueError
            raise value error if the mode is not 'train' or 'eval'
        """
        # transform
        resize = (data_config.img_size[0], data_config.img_size[1])
        color_mean = tuple(data_config.color_mean)
        color_std = tuple(data_config.color_std)
        transform = DataTransform(resize, color_mean, color_std, mode)

        # dataset
        dataset = Dataset(img_list, lbl_list, transform)

        # dataloader
        if mode == 'train':
            return data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        elif mode == 'eval':
            return data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        else:
            raise ValueError('the mode should be train or eval. this mode is not supported')
