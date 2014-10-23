'''
Created on Apr 25, 2014

@author: sstober
'''

import os;
import sys;
from deepthought.experiments.ismir2014.util import load_config;
from deepthought.util.config_util import merge_params;

from deepthought.experiments.ismir2014.train_convnet import train_convnet
# from deepthought.experiments.ismir2014.cross_trial_test import full_cross_trial_test;

def run(params):
    try:
        train_convnet(params);
    except:
        print "Unexpected error:", sys.exc_info()[0]

if __name__ == '__main__':
    config = load_config(default_config=
                 os.path.join(os.path.dirname(__file__),'train_convnet.cfg'), reset_logging=False);                 
    
    batch_subjects = config.get('batch_subjects', xrange(13));
     
    # per person
    for i in batch_subjects:
        hyper_params = { 
                    'experiment_root' : os.path.join(config.experiment_root, 'individual', 'subj'+str(i+1)),
                    'subjects' : [i]
                    };
     
        params = merge_params(config, hyper_params);
 
        run(params);
     
#     # run cross_trial_test
#     try:
#         params = merge_params(config, {});
#         cross_trial_test(params); # Note: this modifies config!
#     except:
#         print "Unexpected error:", sys.exc_info()[0]
#         
#     # for slow rhythms
#     hyper_params = { 
#                     'experiment_root' : os.path.join(config.experiment_root, 'slow'),
#                     'subjects' : [0,1,2,6,7,8]  
#                     };
#     
#     params = merge_params(config, hyper_params);
#     run(params);
#     
#     
#     # for fast rhythms
#     hyper_params = { 
#                     'experiment_root' : os.path.join(config.experiment_root, 'fast'),
#                     'subjects' : [3,4,5,9,10,11,12]  
#                     };
#     
#     params = merge_params(config, hyper_params);
#     run(params);
#     
#     # 24 classes on subject 2
#     hyper_params = { 
#                     'experiment_root' : os.path.join(config.experiment_root, '24subj2'),
#                     'subjects' : [1],
#                     'label_mode' : 'rhythm',
#                     'n_classes' : 24,
#                     };
#     
#     params = merge_params(config, hyper_params);
#     run(params);
#     
#     # one for all
#     hyper_params = { 
#                     'experiment_root' : os.path.join(config.experiment_root, 'all_subjects'),
#                     'subjects' : [0,1,2,3,4,5,6,7,8,9,10,11,12]  
#                     };
#     
#     params = merge_params(config, hyper_params);
#     run(params);