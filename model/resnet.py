# -*- coding: utf-8 -*-
"""ResNet18 model"""

import torch
from torchvision.models as models

from dataloader import DataLoader

from .base_model import BaseModel
from .common.device import setup_device
from .common.criterion import make_criterion
from .common.optimizer import make_optimizer


class ResNet(BaseModel):
    """ResNet Model Class"""

    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.model_name = self.config.model.name
        self.n_classes = self.config.model.n_classes
        self.batch_size = self.config.train.batch_size
        self.n_gpus = self.config.train.n_gpus
        self.resume = self.config.train.resume
        self.metrics = None

        # dataloader
        self.trainloader = None
        self.valloader = None
        self.testloader = None

    def load_data(self, eval: bool):
        """Loads and Preprocess data"""
        # train
        if not eval:
            # train data
            self.train_img_list, self.train_lbl_list = DataLoader().load_data(self.config.data.dataroot, self.config.data.labelroot.train)
            self.trainloader = DataLoader().preprocess_data(data_config, self.train_img_list, self.train_lbl_list, self.batch_size, 'train')

            # val data
            self.val_img_list, self.val_lbl_list = DataLoader().load_data(self.config.data.dataroot, self.config.data.labelroot.val)
            self.valloader = DataLoader().preprocess_data(data_config, self.val_img_list, self.val_lbl_list, self.batch_size, 'eval')
        
        # evaluation
        if eval:
            self.test_img_list, self.test_lbl_list = DataLoader().load_data(self.config.data.dataroot, self.config.data.labelroot.test)
            self.testloader = DataLoader().preprocess_data(data_config, self.test_img_list, self.test_lbl_list, self.batch_size, 'eval')

    def build(self):
        """ Builds model """
        kwargs = {'num_classes': self.n_classes}

        if self.model_name == 'resnet18':
            self.model = models.resnet18(pretrained=False, **kwargs)
        elif self.model_name == 'resnet34':
            self.model = models.resnet34(pretrained=False, **kwargs)
        elif self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=False, **kwargs)
        elif self.model_name == 'resnet101':
            self.model = models.resnet101(pretrained=False, **kwargs)
        elif self.model_name == 'resnet152':
            self.model = models.resnet152(pretrained=False, **kwargs)
        else:
            raise ValueError('This model name is not supported.')

    def _set_training_parameters(self):
        """Sets training parameters"""
        self.epochs = self.config.train.epochs
        self.save_ckpt_interval = self.config.train.save_ckpt_interval
        self.optimizer = make_optimizer(self.model, self.config.train.optimizer)
        self.criterion = make_criterion(self.config.train.criterion)
        
    def train(self):
        """Compiles and trains the model"""
        self.device = setup_device(self.n_gpus)
        self._set_training_parameters()
        self.metrics = None

        train_parameters = {
            'device': self.device,
            'model': self.model,
            'data_loaders': (self.trainloader, self.valloader),
            'epochs': self.epochs,
            'optimizer': self.optimizer,
            'criterion': self.criterion,
            'metrics': self.metrics,
            'save_ckpt_interval': self.save_ckpt_interval,
        }

        # trainer = Trainer(**self.train_parameters)
        # trainer.train()

    def evaluate(self):
        """Predicts resuts for the test dataset"""
        self.device = setup_device(self.n_gpus)
        self.metrics = None

        eval_parameters = {
            'device': self.device,
            'model': self.model,
            'data_loaders': (self.trainloader, self.testloader),
            'epochs': None,
            'optimizer': None,
            'criterion': None,
            'metrics': self.metrics,
            'save_ckpt_interval': None,
        }

        # trainer = Trainer(**self.eval_parameters)
        # trainer.eval()