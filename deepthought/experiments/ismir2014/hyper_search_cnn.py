'''
Created on Apr 25, 2014

@author: sstober
'''

import os;
import sys;
import logging;
log = logging.getLogger(__name__);

from deepthought.experiments.ismir2014.util import load_config;
from deepthought.util.config_util import merge_params;

from deepthought.experiments.ismir2014.train_convnet import train_convnet

# from deepthought.experiments.ismir2014.plot import plot2;

# from concurrent.futures import ThreadPoolExecutor;
# NOTE: ProcessPoolExecutor does not work with GPU / CUDA

def dummy():
    for x in xrange(10000):
        x**x;
    log.info('done');
    
def run(params):
    try:
        log.debug('running {}'.format(params.experiment_root));
#         dummy();
        train_convnet(params);
#         plot2(config.experiment_root);
        
    except:
        log.fatal("Unexpected error:", sys.exc_info());


if __name__ == '__main__':
    config = load_config(default_config=
                 os.path.join(os.path.dirname(__file__),'train_fftconvnet.cfg'), reset_logging=True);                 
    
    lr_values = config.get('lr_values', [0.001, 0.0033, 0.01, 0.00033, 0.033, 0.1]);
    beat_patterns = config.get('beat_patterns', [10,20,30]);
    bar_patterns = config.get('bar_patterns', [10,20,30]);
    beat_pools = config.get('beat_pools', [1,3,5]);
    bar_pools = config.get('bar_pools', [1,3,5]);
    
#     with ThreadPoolExecutor(max_workers=config.num_processes) as executor:
    for lr in lr_values:
        for h1pat in bar_patterns:
            for h1pool in bar_pools:
                for h0pat in beat_patterns:
                    for h0pool in beat_pools:
                    
                        # collect params 
                        exp_str = 'lr{}/h1pool{}/h1pat{}/h0pool{}/h0pat{}'.format(lr,h1pool,h1pat,h0pool,h0pat);
                        hyper_params = { 
                            'experiment_root' : os.path.join(config.experiment_root, exp_str),
                            'learning_rate' : lr,
                            'beat_pool_size' : h0pool,
                            'num_beat_patterns' : h0pat,
                            'bar_pool_size' : h1pool,
                            'num_bar_patterns' : h1pat,
                            };                                                
                        params = merge_params(config, hyper_params);

                        # check if directory already exists
                        if os.path.exists(os.path.join(params.experiment_root, 'epochs')):
                            print 'Already done: '+params.experiment_root;
                            continue;                        
                            
#                             executor.submit(run, params);
                        run(params);
    