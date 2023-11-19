from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import yaml
import os


def load_config(file_path='src/configs/rec_icdar15.yml'):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: config
    """
    ext = os.path.splitext(file_path)[1]
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    with open(file_path, 'rb') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    return config

config = load_config()