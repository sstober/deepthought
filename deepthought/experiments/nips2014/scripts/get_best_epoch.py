#!/usr/bin/env python
'''
Created on Jun 4, 2014

@author: sstober
'''
import itertools;
import logging;

from pylearn2.utils import serial;

from deepthought.util.config_util import init_logging;
from deepthought.experiments.ismir2014.extract_results import _extract_best_results, _get_best_epochs;

if __name__ == '__main__':
    
    init_logging(pylearn2_loglevel=logging.INFO);
    
    # using cached result
    model_file = 'mlp.pkl';
    model = serial.load(model_file);
            
    
    channels = model.monitor.channels;
    
    # directly analyze the model from the train object   
    best_results = _extract_best_results(
                                         channels=channels,
                                         mode='misclass', 
                                         check_dataset='valid',
                                         check_channels=['_y_misclass'],
                                         );    
    best_epochs = _get_best_epochs(best_results);
    best_epoch = best_epochs[-1]; # take last entry -> more stable???
    
    datasets = ['train', 'valid', 'test', 'post'];
    measures = ['_y_misclass', '_objective', '_nll'];
    
    print 'results for epoch {}'.format(best_epoch);
    for measure,dataset in itertools.product(measures,datasets):
        channel = dataset+measure;
        if channel in channels:
            value = float(channels[channel].val_record[best_epoch]);
            print '{:>30} : {:.4f}'.format(channel, value);