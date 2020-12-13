"""Inferrer"""

import torch

from utils.config import Config
from utils.load import load_yaml
from model import get_model
from model.common.device import setup_device
from dataloader.transform import DataTransform


class Inferrer:
    def __init__(self, configfile: str):
        # Config
        config = load_yaml(configfile)
        self.config = Config.from_json(config)

        # Builds model
        self.model = get_model(config)
        self.model.build()

        # device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.model.to(self.device)
        self.model.model.eval()
                
    def preprocess(self, image):
        """Preprocess Image
        PIL.Image to Tensor
        """
        resize = (self.config.data.img_size[0], self.config.data.img_size[1])
        color_mean = tuple(self.config.data.color_mean)
        color_std = tuple(self.config.data.color_std)
        transform = DataTransform(resize, color_mean, color_std, mode='eval')
        image = transform(image).unsqueeze(0) # torch.Size([1, 3, img_size[0], img_size[1]])
        return image

    def infer(self, image=None):
        """Infer an image

        Parameters
        ----------
        image : PIL.Image, optional
            input image, by default None

        Returns
        -------
        int
            class label
        """
        shape = image.size

        tensor_image = self.preprocess(image)

        tensor_image = tensor_image.to(self.device)
        output = self.model.model(tensor_image)
        pred = output.argmax(axis=1)
        pred = pred.cpu().detach().clone()[0].item()

        return pred