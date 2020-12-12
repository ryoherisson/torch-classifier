# -*- coding: utf-8 -*-
"""ResNet18 model"""
from torchvision.models as models

from utils.paths import Paths
from utils.logger import setup_logger, get_logger
from dataloader import DataLoader
from metric import Metric

from .base_model import BaseModel
from .common.device import setup_device, data_parallel
from .common.criterion import make_criterion
from .common.optimizer import make_optimizer
from .common.ckpt import load_ckpt

LOG = get_logger(__name__)

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

        # dataloader
        self.trainloader = None
        self.valloader = None
        self.testloader = None

        # paths
        self.paths = Paths.make_dirs(self.config)
    
        # setup logger
        setup_logger(str(paths.logdir / 'info.log'))

    def load_data(self, eval: bool):
        """Loads and Preprocess data"""
        LOG.info(f'\nLoading {self.config.data.dataroot} dataset...')
        # train
        if not eval:
            # train data
            LOG.info(f'Train data...')
            self.train_img_list, self.train_lbl_list = DataLoader().load_data(self.config.data.dataroot, self.config.data.labelroot.train)
            self.trainloader = DataLoader().preprocess_data(data_config, self.train_img_list, self.train_lbl_list, self.batch_size, 'train')

            # val data
            LOG.info(f'Validation data...')
            self.val_img_list, self.val_lbl_list = DataLoader().load_data(self.config.data.dataroot, self.config.data.labelroot.val)
            self.valloader = DataLoader().preprocess_data(data_config, self.val_img_list, self.val_lbl_list, self.batch_size, 'eval')
        
        # evaluation
        if eval:
            LOG.info(f'Test data...')
            self.test_img_list, self.test_lbl_list = DataLoader().load_data(self.config.data.dataroot, self.config.data.labelroot.test)
            self.testloader = DataLoader().preprocess_data(data_config, self.test_img_list, self.test_lbl_list, self.batch_size, 'eval')

    def build(self):
        """ Builds model """
        kwargs = {'num_classes': self.n_classes}
        pretrained = self.config.model.pretrained

        if self.model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained, **kwargs)
        elif self.model_name == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained, **kwargs)
        elif self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained, **kwargs)
        elif self.model_name == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained, **kwargs)
        elif self.model_name == 'resnet152':
            self.model = models.resnet152(pretrained=pretrained, **kwargs)
        else:
            raise ValueError('This model name is not supported.')

        LOG.info('\n Model was successfully build.')

    def _set_model_parameters(self):
        """Sets model parameters"""
        # CPU or GPU(single, multi)
        self.device = setup_device(self.n_gpus)
        self.model = self.model.to(self.device)
        if self.n_gpus > 1:
            self.model = data_parallel(self.model)

    def _set_training_parameters(self):
        """Sets training parameters"""
        self.epochs = self.config.train.epochs
        self.save_ckpt_interval = self.config.train.save_ckpt_interval
        self.optimizer = make_optimizer(self.model, self.config.train.optimizer)
        self.criterion = make_criterion(self.config.train.criterion)

    def train(self):
        """Compiles and trains the model"""
        LOG.info('\n Training started.')
        self._set_model_parameters()
        self._set_training_parameters()
        
        # load checkpoint
        if self.resume:
            self.model = load_ckpt(self.model, self.resume)            

        self.metrics = Metric(self.n_classes, self.classes, self.paths.metric_dir)

        train_parameters = {
            'device': self.device,
            'model': self.model,
            'data_loaders': (self.trainloader, self.valloader),
            'epochs': self.epochs,
            'optimizer': self.optimizer,
            'criterion': self.criterion,
            'metrics': self.metrics,
            'save_ckpt_interval': self.save_ckpt_interval,
            'ckpt_dir': self.paths.ckpt_dir,
            'summary_dir': self.paths.summary_dir,
        }

        trainer = Trainer(**self.train_parameters)
        trainer.train()

    def evaluate(self):
        """Predicts resuts for the test dataset"""
        LOG.info('\n Prediction started...')

        self._set_model_parameters()

        # load checkpoint
        if self.resume:
            self.model = load_ckpt(self.model, self.resume)            

        self.metrics = Metric(self.n_classes, self.classes, self.paths.metric_dir)

        eval_parameters = {
            'device': self.device,
            'model': self.model,
            'dataloaders': (self.trainloader, self.testloader),
            'epochs': None,
            'optimizer': None,
            'criterion': None,
            'metrics': self.metrics,
            'save_ckpt_interval': None,
            'ckpt_dir': self.paths.ckpt_dir,
            'summary_dir': self.paths.summary_dir,
        }

        trainer = Trainer(**self.eval_parameters)
        trainer.eval()