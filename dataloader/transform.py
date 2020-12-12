# -*- coding: utf-8 -*-
"""DataTransform class"""

from typing import Tuple

import torchvision.transforms as transforms


class DataTransform():
    def __init__(self, resize: Tuple, color_mean: Tuple, color_std: Tuple, mode: str):
        """Transform class to preprocess data

        Parameters
        ----------
        resize : Tuple
            height and width of the image e.g. (128, 128)
        color_mean : Tuple
            mean values of image channels e.g. (0.4914, 0.4822, 0.4465)
        color_std : Tuple
            standard deviation of image channels e.g. (0.2023, 0.1994, 0.2010) 
        mode : String
            train or eval. eval for inference version.

        Raises
        ------
        ValueError
            raise value error if the mode is not 'train' or 'eval'
        """
        if mode == 'train':
            self.transform = transforms.Compose([
                             transforms.Resize(resize),
                             transforms.ToTensor(),
                             transforms.Normalize(color_mean, color_std)
                             ])
        elif mode == 'eval':
            self.transform = transforms.Compose([
                             transforms.Resize((img_size, img_size)),
                             transforms.ToTensor(),
                             transforms.Normalize(color_mean, color_std)
                             ])
        else:
            raise ValueError('datatransforms: this mode is not supported')

    def __call__(self, img):
        return self.transform(img)