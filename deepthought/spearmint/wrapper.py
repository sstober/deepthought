'''
Created on May 11, 2014

@author: sstober
'''
import traceback
import random
import logging;

import os;

log = logging.getLogger(__name__);

import itertools

from pylearn2.utils import serial;
from pylearn2.utils.timing import log_timing

from deepthought.experiments.ismir2014.extract_results import _extract_best_results, _get_best_epochs;

from deepthought.util.config_util import load_config_file, init_logging, merge_params;
from deepthought.util.fs_util import ensure_dir_exists, symlink, touch;
from deepthought.util.yaml_util import flatten_yaml, save_yaml_file, load_yaml;
from deepthought.util.class_util import load_class;

def structural_param_check(params, raise_error=False):
    if params.hop_size > params.input_length:
        msg = 'hop size must not be greater than input length: {} > {}' \
                         .format(params.hop_size, params.input_length);
        if raise_error:
            raise ValueError(msg);
        else:
            log.error(msg);
            return False;
                         
    '''
    #input_length : 
        $beat_pattern_width + 
        $bar_pattern_width -1 + 
        $rhyth_pattern_width -1 + 
        
        $beat_pool_size -1 + 
        $bar_pool_size -1 + 
        $rhythm_pool_size -1
    '''
                         
    min_input_length = 1;
    
    for layer in ['h0', 'h1' ,'h2']:
        min_input_length += params.get(layer+'_pattern_width', 1) - 1;
        min_input_length += params.get(layer+'_pool_size', 1) - 1;
        log.debug('min input length for {}: {}'.format(layer, min_input_length));

    if  params.input_length < min_input_length:
        msg = 'input length is smaller than required minimum: {} < {}'\
                         .format(params.input_length, min_input_length);
        if raise_error:
            raise ValueError(msg);
        else:
            log.error(msg);
            return False;
    
    log.info('structural parameters OK');
    return True;

def convert_to_valid_filename(filename):
    return "".join([c for c in filename if not c in [' ']]).rstrip()

def run_job(job_id, meta_job_path, yaml_template_file, base_config_path, hyper_params, cache_path=None):
    
    # ConstrainedGPEIOptChooser requires NaN or inf to recognize constraints
#     BAD_SOLUTION_RETURN_VALUE = np.inf;
    BAD_SOLUTION_RETURN_VALUE = 1;

    # TODO: nice-to-have: make logging a little nicer
    init_logging(pylearn2_loglevel=logging.INFO);
    
    for key, value in hyper_params.items():
        hyper_params[key] = value[0];
        log.debug('{} = {}'.format(key, hyper_params[key]));
    
    base_config = load_config_file(base_config_path);
    
    # fix dataset path
    localizer_class = base_config.get('localizer_class', 
                                      'deepthought.datasets.rwanda2013rhythms.PathLocalizer'); # for compatibility with old code
    localizer = load_class(localizer_class);
    base_config = localizer.localize_config(base_config);

    if not hasattr(base_config, 'random_seed') \
                            and not hasattr(hyper_params, 'random_seed'):
        random_seed = random.randint(0, 100);
        hyper_params['random_seed'] = random_seed;
        log.debug('using random seed {}'.format(random_seed))

    param_str = '';
    for key in sorted(hyper_params.keys()): # deterministic order
        param_str += '_{}_{}'.format(key, hyper_params[key]);
    
    verbose_job_id = str(job_id) + param_str;
    base_config.verbose_job_id = verbose_job_id;
    

    if cache_path is None:
        cache_path = os.path.join(meta_job_path, 'cache');

    job_output_path = os.path.join(meta_job_path, 'output', str(job_id));
    output_path = os.path.join( 
                               cache_path, 
                               convert_to_valid_filename(param_str)
                               );                            

    # check whether cached result already exists
    model = None;
    failed_file = os.path.join(output_path, 'failed');
    if os.path.exists(output_path):
        # create a link to job-id                    
        symlink(output_path, job_output_path, override=True, ignore_errors=True);
        
        # using cached result
        model_file = os.path.join(output_path, 'mlp.pkl');
        if os.path.exists(model_file):    
            try: 
                with log_timing(log, 'loading cached model from {}'.format(model_file)): 
                    model = serial.load(model_file);
                
        
                    channels = model.monitor.channels;
            except Exception as e:
                log.error('unexpected exception loading model from {}: {} \n{}'\
                      .format(model_file, e, traceback.format_exc())); 
        else:
            # if mlp.pkl is missing but mlp-best.pkl is there, then it was a bad configuration
            if os.path.exists(failed_file):
                log.info('cache contains \'failed\' flag'); 
                return BAD_SOLUTION_RETURN_VALUE;
        
    if model is None:
        
#     output_path = os.path.join(
#                                meta_job_path, 
#                                'output', 
#                                convert_to_valid_filename(verbose_job_id)
#                                );    

        # needs to go here to get the internal reference resolved
        base_config.output_path = output_path;     
        
        # sanity check of structural parameters:
        if not structural_param_check(
                                   merge_params(base_config, hyper_params), 
                                   raise_error=False,
                                   ):
            touch(failed_file, mkdirs=True); # set marker
            return BAD_SOLUTION_RETURN_VALUE;
    
        ensure_dir_exists(output_path);
        symlink(output_path, job_output_path, override=True, ignore_errors=True);
    
        
        yaml = flatten_yaml(
                            yaml_file_path = yaml_template_file, 
                            base_config = base_config, 
                            hyper_params = hyper_params,
                            );
    
        save_yaml_file(
                       yaml_str = yaml, 
                       yaml_file_path = os.path.join(output_path, 'train.yaml')
                       );    
        
        with log_timing(log, 'loading yaml for job {}'.format(job_id)):
            train = load_yaml(yaml)[0];
        
        with log_timing(log, 'running job {} '.format(job_id)):   
            try: 
                train.main_loop();
            except Exception as e:
                log.error('unexpected exception during training: {} \n{}'\
                          .format(e, traceback.format_exc()));
                touch(failed_file, mkdirs=True); # set marker
                return BAD_SOLUTION_RETURN_VALUE;
        
        channels = train.model.monitor.channels;

                 
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
    
    print 'results for job {}'.format(job_id);
    for measure,dataset in itertools.product(measures,datasets):
        channel = dataset+measure;
        if channel in channels:
            value = float(channels[channel].val_record[best_epoch]);
            print '{:>30} : {:.4f}'.format(channel, value);
     
#     return float(channels['test_y_misclass'].val_record[best_epoch]);
    return float(channels['valid_y_misclass'].val_record[best_epoch]);