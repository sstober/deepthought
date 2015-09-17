'''
Created on Apr 10, 2014

@author: sstober
'''
import logging;

import os;
log = logging.getLogger(__name__);

import numpy as np; 

from pylearn2.utils.timing import log_timing

from deepthought.experiments.ismir2014.util import load_config;
from deepthought.util.yaml_util import load_yaml_file, save_yaml_file;
from deepthought.experiments.ismir2014.plot import scan_for_best_performance;
from deepthought.experiments.ismir2014.extract_results import extract_results;

def train_convnet(config):
    
    train, yaml_str = load_yaml_file(
                   os.path.join(os.path.dirname(__file__), 'train_convnet_template.yaml'),
                   params=config,
                   );
    
    save_yaml_file(yaml_str, os.path.join(config.experiment_root, 'settings.yaml'));
        
    with log_timing(log, 'training network'):    
        train.main_loop();
        
def get_default_config_path():
    return os.path.join(os.path.dirname(__file__),'train_convnet.cfg');

if __name__ == '__main__':
    config = load_config(default_config=get_default_config_path(), reset_logging=False);

    if not config.get('only_extract_results', False):
        train_convnet(config);
    
    scan_for_best_performance(config.experiment_root, 'valid_y_misclass');
    scan_for_best_performance(config.experiment_root, 'valid_ptrial_misclass_rate')

    values = extract_results(config.experiment_root, mode='misclass');        
            
    print np.multiply(100, [
#                         1 - values['test_y_misclass'],
#                         1 - values['test_wseq_misclass_rate'],
#                         1 - values['test_wtrial_misclass_rate']]);     
               
                1 - values['frame_misclass'],
                1 - values['sequence_misclass'],
                1 - values['trial_misclass']]);
