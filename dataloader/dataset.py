# -*- coding: utf-8 -*-
"""Dataset class"""
from typing import List

from PIL import Image
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, img_list: List, lbl_list: List, transform: object):
        """Dataset class called by DataLoader class

        Parameters
        ----------
        img_list : List
            a list of image path: e.g. ['./images/1.png', './images/2.png', ./images/3.png', ...]
        lbl_list : List
            a list of labels: e.g. [2, 3, 5, ...]
        transform : object
            transform object to preprocess data
        """
        self.img_list = img_list
        self.lbl_list = lbl_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_filepath = self.img_list[index]
        img = Image.open(img_filepath)
        img = self.transform(img)

        lbl = self.lbl_list[index]

        return img, lbl