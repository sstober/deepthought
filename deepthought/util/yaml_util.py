'''
Created on May 10, 2014

@author: sstober
'''
import os

import logging
log = logging.getLogger(__name__)

from pylearn2.config import yaml_parse
from pylearn2.utils.timing import log_timing

from deepthought.util.config_util import merge_params

def flatten_yaml(yaml_file_path, base_config=None, hyper_params=None):
    yaml_template = load_yaml_template(yaml_file_path)

    params = merge_params(
                          default_params = base_config, 
                          override_params = hyper_params
                          )

    yaml = yaml_template % params

    return yaml
    
def load_yaml_template(yaml_file_path):
    with open(yaml_file_path, 'r') as f:
        yaml_template = f.read()
    f.close()
    
    return yaml_template

def load_yaml_file(yaml_file_path, params=None):
    
    return load_yaml(load_yaml_template(yaml_file_path), params)

def load_yaml(yaml_template, params=None):    
    log.debug('params: {}'.format(params))
    
    if params is not None:
        yaml_str = yaml_template % params
    else:
        yaml_str = yaml_template
    log.debug(yaml_str)

    with log_timing(log, 'parsing yaml'):    
        obj = yaml_parse.load(yaml_str)
    
    return obj, yaml_str

def save_yaml_file(yaml_str, yaml_file_path):
    if save_yaml_file is not None:
        with log_timing(log, 'saving yaml to {}'.format(yaml_file_path)):
            save_dir = os.path.dirname(yaml_file_path)
            if save_dir == '':
                save_dir = '.'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with  open(yaml_file_path, 'w') as yaml_file:
                yaml_file.write(yaml_str) 
            yaml_file.close()
