'''
Created on Jun 20, 2014

@author: sstober

Rather ugly hack to deal with local configurations on different machines.
This should not be necessary for future datasets/experiments 
as environment variables will be used instead.
'''

import logging;
log = logging.getLogger(__name__);

import os;
import deepthought;

class PathLocalizer(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.dataset_root_jeep2 = '/Users/sstober/work/datasets/Dan/eeg'
        self.dataset_root_gpu = '/home/likewise-open/UWO/sstober/dev/datasets/Dan/eeg'
        self.dataset_root = os.path.join(deepthought.DATA_PATH, 'rwanda2013rhythms', 'eeg');
        
    def localize_yaml(self, yaml):
        # fix data path        
        log.info('changing data path to: {}'.format(self.dataset_root));

        yaml = yaml.replace(self.dataset_root_jeep2, self.dataset_root);
        yaml = yaml.replace(self.dataset_root_gpu, self.dataset_root);

        return yaml;
    
    def localize_config(self, config):
        # fix dataset path
        config.dataset_root = self.dataset_root;
            
        # fix bar length
        if config.subjects[0] in [0,1,2,6,7,8]: # NOTE: expecting single subject
            # beat length 180ms -> bar length 2160ms -> 2.16 * 400
            config.samples_per_bar = config.get('samples_per_bar_180', 864); 
            config.input_length = config.get('input_length_180', 33);
        else:
            # beat length 240ms -> bar length 2880ms -> 2.88 * 400
            config.samples_per_bar = config.get('samples_per_bar_240', 1152); 
            config.input_length = config.get('input_length_240', 45);
        log.info('using {} samples per bar -> input length {}'.format(config.samples_per_bar,config.input_length));
            
        return config;