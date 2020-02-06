import yaml
import json
import argparse

class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

class Config:
    def get_config_from_yaml(path):
        with open(path, 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
        config = obj(cfg)
        return config

    def get_config_from_json(path):
        with open(path) as fd:
            config = json.load(fd, object_hook=dict_to_sns)
        return config
