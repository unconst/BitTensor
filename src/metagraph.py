from loguru import logger
import sys

class Metagraph():
    def __init__(self, config):
        self.config = config
        self.remote_neurons = []
