#!/usr/bin/env python
'''
Created on May 20, 2014

@author: sstober
'''

import os
import argparse

import logging;
log = logging.getLogger(__name__);

from deepthought.util.config_util import init_logging;
from deepthought.util.fs_util import save;
from deepthought.pylearn2ext.util import process_dataset;

from deepthought.experiments.nips2014.scripts.generate_plots import load_results;

from pylearn2.utils.timing import log_timing

def extract_output(experiment_root):
    train, model = load_results(experiment_root);
        
    # get the datasets with their names from the monitor
    for key, dataset in train.algorithm.monitoring_dataset.items():
        # process each dataset 
        with log_timing(log, 'processing dataset \'{}\''.format(key)): 
            y_real, y_pred, output = process_dataset(model, dataset)
            
            save(os.path.join(experiment_root, 'cache', key+'_output.pklz'), (y_real, y_pred, output));    

if __name__ == '__main__':
    init_logging(pylearn2_loglevel=logging.INFO);
    parser = argparse.ArgumentParser(prog='extract_output', 
                                     description='extracts model output and labels for further processing');
     
    # global options
    parser.add_argument('path', help='root path of the experiment');
     
    args = parser.parse_args();
           
    experiment_root = args.path;

#     experiment_root = '/Users/sstober/git/deepbeat/deepbeat/spearmint/h0_input47/20041_h0_pattern_width_[47]_h0_patterns_[30]_h0_pool_size_[1]_learning_rate_[0.01]'
#     experiment_root = '/Users/sstober/git/deepbeat/deepbeat/spearmint/best/export/1'

    extract_output(experiment_root);
