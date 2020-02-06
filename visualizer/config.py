import yaml
import json
import argparse

from types import SimpleNamespace

def dict_to_sns(d):
    return SimpleNamespace(**d)

class Config:
    def get_config_from_yaml(path):
        with open(path, 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
        config = dict_to_sns(cfg)
        return config

    def get_config_from_json(path):
        with open(path) as fd:
            config = json.load(fd, object_hook=dict_to_sns)
        return config
