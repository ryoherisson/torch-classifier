# -*- coding: utf-8 -*-
"""Transform class"""

from typing import Tuple

import torchvision.transforms as transforms


class DataTransforms():
    def __init__(self, img_size: Tuple, color_mean: Tuple, color_std: Tuple, phase: str):
        """Transform class to preprocess data

        Parameters
        ----------
        img_size : Tuple
            height and width of the image e.g. (128, 128)
        color_mean : Tuple
            mean values of image channels e.g. (0.4914, 0.4822, 0.4465)
        color_std : Tuple
            standard deviation of image channels e.g. (0.2023, 0.1994, 0.2010) 
        phase : String
            train or eval. eval for inference version.

        Raises
        ------
        ValueError
            raise value error if the phase is not 'train' or 'eval'
        """
        if phase == 'train':
            self.transform = transforms.Compose([
                             transforms.Resize((img_size, img_size)),
                             transforms.ToTensor(),
                             transforms.Normalize(color_mean, color_std)
                             ])
        elif phase == 'eval':
            self.transform = transforms.Compose([
                             transforms.Resize((img_size, img_size)),
                             transforms.ToTensor(),
                             transforms.Normalize(color_mean, color_std)
                             ])
        else:
            raise ValueError('datatransforms: this phase is not supported')

    def __call__(self, img):
        return self.transform(img)