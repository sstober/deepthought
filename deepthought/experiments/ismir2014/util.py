'''
Created on Mar 5, 2014

@author: sstober
'''
import warnings;

from sklearn import cross_validation;
import logging;

import argparse;
from config import Config;
import pylearn2.utils.logger;

import deepthought.util.fs_util as fs_util;

def reset_pylearn2_logging(level=logging.WARN):
    logging.info('resetting pylearn2 logging');
    # this reset the whole pylearn2 logging system -> full control handed over    
    pylearn2.utils.logger.restore_defaults();
    logging.getLogger("pylearn2").setLevel(level)

def load_config(default_config='train_sda.cfg', reset_logging=True):
    parser = argparse.ArgumentParser(description='Train a Stacked Denoising Autoencoder');
    parser.add_argument('-c', '--config', help='specify a configuration file');
    parser.add_argument('-v', '--verbose', help='increase output verbosity", action="store_true');    
    args = parser.parse_args();
    
    if args.config == None:
        configfile = default_config; # load default config
    else:
        configfile = args.config;
    config = Config(file(configfile));  
    
    if args.verbose or config.logger.level=='debug':
        loglevel = logging.DEBUG;
    else:
        loglevel = logging.INFO;
    
    logging.basicConfig(format=config.logger.pattern, level=loglevel);    

    if reset_logging or config.get('reset_logging', True):
        reset_pylearn2_logging();
        
    logging.info('using config {0}'.format(configfile));
    
    # disable annoying deprecation warnings
    warnings.simplefilter('once', UserWarning)
    warnings.simplefilter('default')
    
    return config;

def splitdata(dataset, ptest = 0.1, pvalid = 0.1):
    
    data, labels = dataset;       
    ptrain = 1-ptest-pvalid;
    
    data_train, data_temp, labels_train, labels_temp = \
        cross_validation.train_test_split(data, labels, test_size=(1-ptrain), random_state=42);
    data_valid, data_test, labels_valid, labels_test = \
        cross_validation.train_test_split(data_temp, labels_temp, test_size=ptest/(pvalid+ptest), random_state=42);
        
    train_set = (data_train, labels_train);
    valid_set = (data_valid, labels_valid);
    test_set = (data_test, labels_test);
    
    logging.info('Split data into {0} train, {1} validation, {2} test: {3} {4} {5}'.format(ptrain, pvalid, ptest, data_train.shape, data_valid.shape, data_test.shape));
    
    return train_set, valid_set, test_set;

def load(filepath):
    return fs_util.load(filepath);


def save(filepath, data):
    return fs_util.save(filepath, data);
    
    
#     # Load the dataset
#     f = gzip.open(dataset, 'rb')
#     train_set, valid_set, test_set = cPickle.load(f)
#     f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.
