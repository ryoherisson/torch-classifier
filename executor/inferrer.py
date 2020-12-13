"""Inferrer"""

from utils.config import Config


class Inferrer:
    def __init__(self, config):
        self.config = Config.from_json(config)
        
