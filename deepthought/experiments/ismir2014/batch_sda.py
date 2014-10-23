'''
Created on Apr 22, 2014

@author: sstober
'''

import os;
from deepthought.experiments.ismir2014.util import load_config;
from deepthought.util.config_util import merge_params;

from deepthought.experiments.ismir2014.plot import scan_for_best_performance;

from train_sda_mlp import train_mlp;

if __name__ == '__main__':
    config = load_config(default_config=
                 os.path.join(os.path.dirname(__file__),'train_sda_mlp.cfg'), reset_logging=True);
                 
#     # FIXME:
#     config.experiment_root = '/Users/stober/git/deepbeat/deepbeat/output/gpu/sda/exp2.14all/';
    
    
    for i in xrange(13):
        hyper_params = { 
                    'experiment_root' : os.path.join(config.experiment_root, 'subj'+str(i+1)),
                    'subjects' : [i]  
                    # NOTE: layerX_content should still  point to global sda/ folder
                    };
                    
        if config.global_sda == False: 
            hyper_params['layer0_content'] = os.path.join(hyper_params['experiment_root'], 'sda', 'sda_layer0_tied.pkl');
            hyper_params['layer1_content'] = os.path.join(hyper_params['experiment_root'], 'sda', 'sda_layer1_tied.pkl');
            hyper_params['layer2_content'] = os.path.join(hyper_params['experiment_root'], 'sda', 'sda_layer2_tied.pkl');
            hyper_params['layer3_content'] = os.path.join(hyper_params['experiment_root'], 'sda', 'sda_layer3_tied.pkl');
        
        params = merge_params(config, hyper_params);

        if os.path.exists(os.path.join(params.experiment_root, 'epochs')):
            print 'skipping existing path: {}'.format(params.experiment_root);
            continue;

        train_mlp(params);
        
    # generate plot.pdfs
#     plot_batch(config.experiment_root);
    
    # print best peformance values
    for i in xrange(13):
        scan_for_best_performance(os.path.join(config.experiment_root, 'subj'+str(i+1)));
