'''
Created on Apr 25, 2014

@author: sstober
'''

import logging;

import os;
log = logging.getLogger(__name__);

import numpy as np;

from deepthought.experiments.ismir2014.util import load_config;
from deepthought.util.yaml_util import load_yaml_file, save_yaml_file;
from deepthought.util.config_util import merge_params;

from pylearn2.utils.timing import log_timing

from deepthought.experiments.ismir2014.extract_results import extract_best_result, extract_results;

def train_mlp(params):
    train, yaml_str = load_yaml_file(
                   os.path.join(os.path.dirname(__file__), 'cross_trial_template.yaml'),
                   params=params,
                   );
    
    save_yaml_file(yaml_str, os.path.join(params.experiment_root, 'settings.yaml'));
        
    with log_timing(log, 'training network'):    
        train.main_loop();
        
    # TODO: get performance values

def full_cross_trial_test(config):
    
    '''
        00    '180C_10.wav180F_8_180afr.wav',
        01    '180C_10.wav180F_8_180west.wav',
        02    '180C_10.wav180F_9_180afr.wav',
        03    '180C_10.wav180F_9_180west.wav',
        04    '180C_11.wav180F_12_180west.wav',
        05    '180C_11.wav180F_13_180west.wav',
        06    '180C_12.wav180F_11_180west.wav',
        07    '180C_12.wav180F_13_180west.wav',
        08    '180C_13.wav180F_11_180west.wav',
        09    '180C_13.wav180F_12_180west.wav',
        10    '180C_3.wav180F_4_180afr.wav',
        11    '180C_3.wav180F_5_180afr.wav',
        12    '180C_4.wav180F_3_180afr.wav',
        13    '180C_4.wav180F_5_180afr.wav',
        14    '180C_5.wav180F_3_180afr.wav',
        15    '180C_5.wav180F_4_180afr.wav',
        16    '180C_8.wav180F_10_180afr.wav',
        17    '180C_8.wav180F_10_180west.wav',
        18    '180C_8.wav180F_9_180afr.wav',
        19    '180C_8.wav180F_9_180west.wav',
        20    '180C_9.wav180F_10_180afr.wav',
        21    '180C_9.wav180F_10_180west.wav',
        22    '180C_9.wav180F_8_180afr.wav',
        23    '180C_9.wav180F_8_180west.wav',
    '''
    subject = config.subjects[0];
    african = [0,2,10,11,12,13,14,15,16,18,20,22];
    western = [1,3,4,5,6,7,8,9,17,19,21,23];
    
    if subject in [3,4,5,9,10,11,12]: # 240 rhythms
        for i in xrange(12):
            african[i] += 24;
            western[i] += 24;
    
    pairs = [];
    index = []
    for i,a in enumerate(african): 
        for j,w in enumerate(western):
            pairs.append([a,w]);
            index.append([i,j]);
            
    results = pair_cross_trial_test(config, pairs=pairs)[0];
    
    matrix = np.zeros([len(african), len(western), 3])
    for i,r in enumerate(results):
        matrix[index[i][0], index[i][1]] = r;
        
    print matrix[:,:,0];
    print np.mean(results, axis=0);
#     print np.mean(results, axis=1);
    
def pair_cross_trial_test(config, pairs=None):
    
    if pairs is None:
        pairs = [
             [18, 19],
             [20, 21],
             [22, 23],
             [ 0,  1],
             [15,  9],
             [16, 17],
             [11,  5],
             [12,  6],
             [ 2,  3],
             [10,  4],
             [13,  7],
             [14,  8],
             ];

#     config.experiment_root = os.path.join(config.experiment_root, 'cross-trial') ;
    
    accuracy = np.zeros(len(pairs))
    results = np.zeros([len(pairs),3]);
    for i in xrange(len(pairs)):
        
        test_stimulus_ids = pairs[i];
        
        train_stimulus_ids = set();
        for j in xrange(48):
            if not j in test_stimulus_ids:
                train_stimulus_ids.add(j);
        train_stimulus_ids = list(train_stimulus_ids);
                
        log.info('training stimuli: {} \t test stimuli: {}'.format(train_stimulus_ids, test_stimulus_ids));
        
        hyper_params = { 
                    'experiment_root' : os.path.join(config.experiment_root, 'pair'+str(pairs[i])),
                    'remove_train_stimulus_ids' : test_stimulus_ids,
                    'remove_test_stimulus_ids' : train_stimulus_ids, 
                    };
    
        params = merge_params(config, hyper_params);
        
        # dummy values for testing
#         results[i] = [i, i*10, i*100.];     
#         accuracy[i] = 0;
#         continue;
        
        if os.path.exists(os.path.join(params.experiment_root, 'mlp.pkl')):
            print 'found existing mlp.pkl: {}'.format(params.experiment_root);
        else:
            print 'no mlp.pkl found at: {}'.format(params.experiment_root);
            if not config.get('only_extract_results', False):                
                train_mlp(params);  
        
        values = extract_results(params.experiment_root, mode='misclass');        
            
        results[i] = np.multiply(100, [
#                         1 - values['test_y_misclass'],
#                         1 - values['test_wseq_misclass_rate'],
#                         1 - values['test_wtrial_misclass_rate']]);
                   
                    1 - values['frame_misclass'],
                    1 - values['sequence_misclass'],
                    1 - values['trial_misclass']]);
        
        accuracy[i] = 100 * (1 - extract_best_result(params.experiment_root, mode='misclass', check_dataset='test')[0]);
    
    print results;
    print results.mean(axis=0);
    print results.max(axis=1);
    
    print accuracy;
    print accuracy.mean();
    
    return results, accuracy;

if __name__ == '__main__':
    config = load_config(default_config=
                 os.path.join(os.path.dirname(__file__),'train_convnet.cfg'), reset_logging=True);                 
    
    # override settings
#     config.experiment_root = os.path.join(config.experiment_root, 'cross-trial');
#     config.subjects = None;
#     config.subjects = 8; # subj9 0-indexed
#     config.max_epochs = 5;

    # FIXME: remove manual override
#     config.only_extract_results = True;
#     config.experiment_root = '/Users/sstober/git/deepbeat/deepbeat/output/gpu/fftcnn/cross-test2/'
#     config.experiment_root = '/Users/sstober/git/deepbeat/deepbeat/output/gpu/fftcnn/exp08.a-crosstrial.subj9/';
    
    if config.get('full_cross_trial', False):
        full_cross_trial_test(config);
    else:
        pair_cross_trial_test(config);
    
    
#     compare_best_result(config.experiment_root, mode='f1', check_dataset='valid');
#     compare_best_result(config.experiment_root, mode='misclass', check_dataset='valid');
   
    