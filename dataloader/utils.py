"""Data Utils"""

from pathlib import Path

import pandas as pd


def make_data_list(dataroot: str, labelpath: str):
    """Make data list from dataroot and labelroot

    Parameters
    ----------
    dataroot : str
        a path to the image directory
    labelpath : str
        a path to the csv file of labels

    Returns
    -------
    Tuple of list
        img_list: e.g. ['./data/images/car1.png', './data/images/dog4.png', ...]
        lbl_list: e.g. [3, 5, ...]
    """
    # read csv as dataframe
    df = pd.read_csv(labelpath)
    
    # image path list
    filename_list = df['image'].values.tolist()
    img_list = [Path(dataroot) / filename for filename in filename_list]

    # label list
    lbl_list = df['label'].astype('int').values.tolist()

    return (img_list, lbl_list)
