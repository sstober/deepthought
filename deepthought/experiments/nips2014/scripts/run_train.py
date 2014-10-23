#!/usr/bin/env python
'''
Created on Jun 4, 2014

@author: sstober
'''
# import socket;
import logging;
log = logging.getLogger(__name__);
import argparse;
import random;
from deepthought.util.config_util import init_logging, load_config_file, empty_config;
from deepthought.util.yaml_util import load_yaml_template, load_yaml;
from deepthought.util.class_util import load_class;

if __name__ == '__main__':
    
    init_logging(pylearn2_loglevel=logging.INFO);
    
    #     print args
    # parse arguments using optparse or argparse or what have you
    parser = argparse.ArgumentParser(prog='run_train', 
                                     description='run a train algorithm as specified by a YAML file');
                                     
    # global options
    parser.add_argument('yaml', default='train.yaml', help='path of the YAML file to run');
    
    parser.add_argument("-c", "--config", #type=str,
                        help="specify a config file");
                    
    parser.add_argument("-l", "--localizer", #type=str,                        
                        help="specify a custom localizer");
                    
    args = parser.parse_args();
    
    train_yaml = load_yaml_template(args.yaml);
    
    # load optional settings
    if args.config is not None:
        config = load_config_file(args.config);
    else:
        config = empty_config();
        
    if not hasattr(config, 'random_seed'):
        random_seed = random.randint(0, 100);
        config.random_seed = random_seed;
        log.debug('using random seed {}'.format(random_seed))
    
    # load optional localizer
    if args.localizer is not None:
        localizer_class = args.localizer;
    else:
        localizer_class = config.get('localizer_class', 
                                      'deepthought.datasets.rwanda2013rhythms.PathLocalizer'); # for compatibility with old code    
    localizer = load_class(localizer_class);
    
    # localize settings
    config = localizer.localize_config(config);
    
    # apply settings    
    train_yaml = train_yaml % config;
    
    # localize YAML
    train_yaml = localizer.localize_yaml(train_yaml);   
         
    train, _ = load_yaml(train_yaml);
    
    train.main_loop();