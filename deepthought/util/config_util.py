'''
Created on May 10, 2014

@author: sstober
'''

import logging
from config import Config, ConfigMerger, overwriteMergeResolve

import warnings

default_logging_pattern = '%(relativeCreated)d\t %(levelname)s\t ' \
                          '%(process)d %(funcName)s@%(filename)s:%(lineno)d : %(message)s'

def init_logging(loglevel=logging.DEBUG, pylearn2_loglevel=logging.WARN, pattern=default_logging_pattern):
    logging.basicConfig(level=loglevel)
    suppress_warnings()

    reset_pylearn2_logging(pylearn2_loglevel)

    logger = logging.getLogger('deepthought')

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter(pattern)

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

def suppress_warnings():
    # disable annoying deprecation warnings
    warnings.simplefilter('once', UserWarning)
    warnings.simplefilter('default')

def empty_config():
    return Config()

def load_config_file(config_file_path):
    return Config(file(config_file_path))

def merge_params(default_params, override_params):
    merger = ConfigMerger(resolver=overwriteMergeResolve)
    params = Config()
    if default_params is not None:
        merger.merge(params, default_params)
    if override_params is not None:
        merger.merge(params, override_params)
    return params

def reset_pylearn2_logging(level=logging.WARN):
    logging.info('resetting pylearn2 logging')
    import pylearn2.utils.logger
    # this reset the whole pylearn2 logging system -> full control handed over    
    pylearn2.utils.logger.restore_defaults()
    logging.getLogger("pylearn2").setLevel(level)

def setup_logger(config):
    if config.level=='debug':
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
        
    if config.get('reset_pylearn2_logging', True):
        reset_pylearn2_logging()
        
    logging.basicConfig(format=config.pattern, level=loglevel)