'''
Created on May 6, 2014

@author: sstober
'''

import traceback;

import os;
import numpy as np;

from deepthought.experiments.ismir2014.util import load_config, save;
from deepthought.util.config_util import merge_params;
from deepthought.experiments.ismir2014.train_convnet import train_convnet, get_default_config_path
from deepthought.experiments.ismir2014.extract_results import extract_results, extract_output;


def run_experiment(config, hyper_params, random_seeds):
    
    experiment_root = hyper_params['experiment_root'];
    
    best_acc = -1;
    best_results = [np.NAN, np.NAN, np.NAN];
    for seed in random_seeds:
        hyper_params['random_seed'] = seed;
        hyper_params['experiment_root'] = experiment_root + '.' + str(seed);            
    
        params = merge_params(config, hyper_params);
    
        if os.path.exists(os.path.join(params.experiment_root, 'mlp.pkl')):
            print 'found existing mlp.pkl: {}'.format(params.experiment_root);
        else:
            print 'no mlp.pkl found at: {}'.format(params.experiment_root);
            if not config.get('only_extract_results', False):
                train_convnet(params);
        
        try:
            values = extract_results(params.experiment_root, mode='misclass');        
            
            results = np.multiply(100, [
#                         1 - values['test_y_misclass'],
#                         1 - values['test_wseq_misclass_rate'],
#                         1 - values['test_wtrial_misclass_rate']]);     
                       
                        1 - values['frame_misclass'],
                        1 - values['sequence_misclass'],
                        1 - values['trial_misclass']]);           
            
            # save the best results
            if np.max(results[2]) > best_acc:
                best_results = results; 
                best_acc = np.max(results[2]);
        except:
            print traceback.format_exc();
            results = [np.NAN, np.NAN, np.NAN];
            
        print 'results for seed {}: {}'.format(seed, results);
        
        if params.save_output:
            output = extract_output(params, values['best_epoch']);
            save(os.path.join(params.experiment_root, 'best_output.pklz'), output);
        
    print 'best results: {}'.format(best_results);
    return best_results;

def run_experiments_for_table_row(config):
        
    table_row = np.zeros(14);
    
    random_seeds = config.get('random_seeds', [config.random_seed]);
    config.random_seed = None; # reset to allow override
    
    # individual models
    individual_acc = np.zeros((13,3));
    for i in xrange(13):
        hyper_params = { 
                'experiment_root' : os.path.join(config.experiment_root, 'subj{}'.format(i+1)),
                'subjects' : [i],  
                # NOTE: layerX_content should still  point to global sda/ folder
                };            
        individual_acc[i] = run_experiment(config, hyper_params, random_seeds);                                
    print 'individual acc: \n{}'.format(individual_acc);
    table_row[2:5] = np.mean(individual_acc, axis=0);
    print table_row[2:5];
    
    groups = [
              [5, '180', [0,1,2,6,7,8]],
              [8, '240', [3,4,5,9,10,11,12]],
              [11, 'all', 'all']
    ];
    
    for col, group, subjects in groups:
        hyper_params = { 
                    'experiment_root' : os.path.join(config.experiment_root, str(group)),
                    'subjects' : subjects
                    # NOTE: layerX_content should still  point to global sda/ folder
                    };
        table_row[col:col+3] = run_experiment(config, hyper_params, random_seeds);
        
    print table_row;
    return table_row;

if __name__ == '__main__':
    config = load_config(default_config=get_default_config_path());
    
    ## this will save the raw network output for the best model 
#     config.save_output = True;
    
    # FIXME: remove manual override
#     config.only_extract_results = True;
#     
#     config.experiment_root = '/Users/sstober/git/deepbeat/deepbeat/output/gpu.old/cnn/bigbatch/';
#     
#     config.experiment_root = '/Users/sstober/git/deepbeat/deepbeat/output/gpu/fftcnn/exp05.2/';  ## <= BEST so far
    # 72.2 & 72.8 & 73.7
    
    row = run_experiments_for_table_row(config);
    
    print config.experiment_root;
    s = '';
    for i,f in enumerate(row): 
        if i < 2:
            s += '& {:.2f} '.format(f);
        else:
            s += '& {:.1f} '.format(f);
        if i in [1, 4, 7, 10]:
            s += '&';
    print s;